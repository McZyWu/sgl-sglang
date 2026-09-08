import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDPAttnSchedulerMetadata(CustomTestCase):
    @staticmethod
    def _batch(mode, bs=4):
        return SimpleNamespace(
            forward_mode=mode,
            batch_size=lambda: bs,
            extend_num_tokens=bs * 8,
            extend_lens=[8] * bs,
            extend_logprob_start_lens=[7] * bs,
            return_logprob=False,
        )

    @contextmanager
    def _fused_adapter(self, dp_size=1):
        tbo_preparer = Mock()
        tbo_preparer.prepare_all_gather.side_effect = lambda batch: (
            False,
            (batch.forward_mode if batch is not None else ForwardMode.IDLE).value,
        )
        tbo_preparer.compute_output.return_value = (None, None)
        adapter = dp_attn.SchedulerDPAttnAdapter(
            model_runner=SimpleNamespace(
                prefill_cuda_graph_runner=None,
                spec_algorithm=SpeculativeAlgorithm.DSPARK,
                model_config=object(),
            ),
            tp_group=SimpleNamespace(
                device_group=object(), device="cpu", cpu_group=object()
            ),
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            offload_tags=set(),
            ps=SimpleNamespace(attn_tp_size=4, attn_cp_size=1),
            model_config=object(),
            enable_overlap=True,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            get_require_mlp_sync=lambda: True,
        )
        with (
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo_preparer),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(dp_attn, "cuda_graph_fully_disabled", return_value=False),
            patch.object(dp_attn, "require_mlp_tp_gather", return_value=dp_size > 1),
            patch.object(
                dp_attn,
                "get_parallel",
                return_value=SimpleNamespace(dp_size=dp_size, dwdp_size=1),
            ),
            patch.object(
                dp_attn,
                "get_schedule",
                return_value=SimpleNamespace(disable_overlap_schedule=False),
            ),
            patch.object(
                dp_attn.SchedulerDPAttnAdapter,
                "get_idle_batch",
                side_effect=lambda: self._batch(ForwardMode.IDLE, 0),
            ),
        ):
            yield adapter

    def test_dp1_fused_prefill_wins_over_decode(self):
        prefill = self._batch(ForwardMode.EXTEND)
        decode = self._batch(ForwardMode.DECODE)
        with (
            self._fused_adapter() as adapter,
            patch.object(dp_attn.MLPSyncBatchInfo, "all_gather") as gather,
        ):
            batch, has_prefill, probe_valid = (
                adapter.prepare_speculative_prefill_or_decode_batch(
                    prefill, decode, False
                )
            )
        self.assertIs(batch, prefill)
        self.assertTrue(has_prefill)
        self.assertTrue(probe_valid)
        self.assertTrue(batch.is_extend_in_batch)
        self.assertEqual(batch.global_num_tokens, [32])
        self.assertEqual(decode.forward_mode, ForwardMode.DECODE)
        gather.assert_not_called()

    def test_dp1_fused_invalid_probe_requires_metadata_refresh(self):
        # Finished/retracted requests invalidate the pre-update decode probe.
        # Returning True here skips the post-update sync and retains stale
        # is_extend_in_batch from prefill, misrouting DSpark into _forward_prefill.
        with (
            self._fused_adapter() as adapter,
            patch.object(dp_attn.MLPSyncBatchInfo, "all_gather") as gather,
        ):
            batch, has_prefill, probe_valid = (
                adapter.prepare_speculative_prefill_or_decode_batch(None, None, False)
            )
        self.assertIsNone(batch)
        self.assertFalse(has_prefill)
        self.assertFalse(probe_valid)
        gather.assert_not_called()

    def test_dp1_fused_decode_clears_prefill_metadata_and_restores_verify_mode(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                decode = self._batch(mode)
                decode.is_extend_in_batch = True
                with (
                    self._fused_adapter() as adapter,
                    patch.object(dp_attn.MLPSyncBatchInfo, "all_gather") as gather,
                ):
                    batch, has_prefill, probe_valid = (
                        adapter.prepare_speculative_prefill_or_decode_batch(
                            None, decode, True
                        )
                    )
                self.assertIs(batch, decode)
                self.assertEqual(batch.forward_mode, mode)
                self.assertFalse(has_prefill)
                self.assertTrue(probe_valid)
                self.assertFalse(batch.is_extend_in_batch)
                self.assertEqual(batch.global_num_tokens, [4])
                self.assertTrue(batch.can_run_decode_cuda_graph)
                gather.assert_not_called()

    def test_dp1_fused_idle_emits_no_batch(self):
        with self._fused_adapter() as adapter:
            batch, has_prefill, probe_valid = (
                adapter.prepare_speculative_prefill_or_decode_batch(None, None, True)
            )
        self.assertIsNone(batch)
        self.assertFalse(has_prefill)
        self.assertTrue(probe_valid)

    def test_dp2_fused_peer_prefill_idles_local_decode(self):
        decode = self._batch(ForwardMode.TARGET_VERIFY)

        def gather_peer_prefill(output, local, group):
            rows = output.view(2, 4, -1)
            rows[0] = local
            # Peer prefill: 16 tokens, two requests, no decode graph, EXTEND.
            rows[1] = torch.tensor([16, 2, 0, 1, 0, ForwardMode.EXTEND.value, 0, 1, 1])

        with (
            self._fused_adapter(dp_size=2) as adapter,
            patch.object(
                dp_attn,
                "get_tp_group",
                return_value=SimpleNamespace(active_ranks_cpu=torch.ones(8)),
            ),
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                side_effect=gather_peer_prefill,
            ) as gather,
        ):
            batch, has_prefill, probe_valid = (
                adapter.prepare_speculative_prefill_or_decode_batch(None, decode, True)
            )
        self.assertIsNot(batch, decode)
        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
        self.assertEqual(decode.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertTrue(has_prefill)
        self.assertTrue(probe_valid)
        self.assertTrue(batch.is_extend_in_batch)
        self.assertEqual(batch.global_num_tokens, [0, 16])
        self.assertEqual(batch.global_num_tokens_for_logprob, [0, 2])
        gather.assert_called_once()

    def test_skip_all_gather_policy(self):
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(dp_size=1))
            self.assertFalse(dp_attn.should_skip_scheduler_all_gather(dp_size=2))
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(dp_size=2))

    def test_dp1_skip_preserves_local_tbo_metadata(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 4,
        )
        tbo_preparer = Mock()
        tbo_preparer.prepare_all_gather.return_value = (
            True,
            ForwardMode.DECODE.value,
        )
        tbo_preparer.compute_output.return_value = (2, ForwardMode.DECODE)

        with (
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo_preparer),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(dp_attn.MLPSyncBatchInfo, "all_gather") as all_gather,
        ):
            result = dp_attn.prepare_mlp_sync_batch_raw(
                batch,
                model_runner=SimpleNamespace(
                    prefill_cuda_graph_runner=None,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    model_config=object(),
                ),
                dp_size=1,
                attn_tp_size=4,
                attn_cp_size=1,
                tp_group=SimpleNamespace(
                    device_group=object(), device="cpu", cpu_group=object()
                ),
                get_idle_batch=Mock(
                    side_effect=AssertionError("DP1 must not emit idle batch")
                ),
                disable_cuda_graph=False,
                require_mlp_tp_gather=False,
                disable_overlap_schedule=True,
                offload_tags=set(),
            )

        all_gather.assert_not_called()
        self.assertEqual(result.global_num_tokens, [4])
        self.assertEqual(result.tbo_split_seq_index, 2)
        self.assertEqual(result.global_forward_mode, ForwardMode.DECODE)
        self.assertEqual(result.recv_skipper_forward_mode, ForwardMode.DECODE)
        self.assertEqual(
            tbo_preparer.compute_output.call_args.args[0].tolist(),
            [[1, ForwardMode.DECODE.value]],
        )


if __name__ == "__main__":
    unittest.main()
