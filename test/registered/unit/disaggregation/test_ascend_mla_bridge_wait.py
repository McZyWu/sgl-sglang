"""CPU tests for the bounded NPU MLA scatter completion wait."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.ascend.conn import _wait_for_npu_mla_bridge_chunk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAscendMLABridgeWait(CustomTestCase):
    def run_wait(self, results, *, deadline=10):
        resources = SimpleNamespace(
            finish_chunk=unittest.mock.Mock(side_effect=results)
        )
        with (
            patch("sglang.srt.disaggregation.ascend.conn.time.sleep") as sleep,
            patch(
                "sglang.srt.disaggregation.ascend.conn.time.monotonic", return_value=0
            ),
        ):
            _wait_for_npu_mla_bridge_chunk(
                resources,
                room=1,
                generation=2,
                session="s",
                layer=3,
                slot=0,
                lease_id=4,
                deadline=deadline,
            )
        return resources.finish_chunk, sleep

    def test_first_query_ready_does_not_sleep(self):
        resources = SimpleNamespace(finish_chunk=unittest.mock.Mock(return_value=True))
        with (
            patch(
                "sglang.srt.disaggregation.ascend.conn.time.monotonic",
                side_effect=AssertionError("ready query must not inspect deadline"),
            ) as monotonic,
            patch("sglang.srt.disaggregation.ascend.conn.time.sleep") as sleep,
        ):
            _wait_for_npu_mla_bridge_chunk(
                resources,
                room=1,
                generation=2,
                session="s",
                layer=3,
                slot=0,
                lease_id=4,
                deadline=-1,
            )
        query = resources.finish_chunk
        self.assertEqual(query.call_count, 1)
        query.assert_called_once_with(
            room=1, generation=2, session="s", layer=3, slot=0, lease_id=4
        )
        monotonic.assert_not_called()
        sleep.assert_not_called()

    def test_false_queries_yield_four_times_then_poll_at_one_ms(self):
        for count in (1, 4, 5):
            with self.subTest(count=count):
                query, sleep = self.run_wait([False] * count + [True])
                self.assertEqual(query.call_count, count + 1)
                self.assertEqual(
                    [call.args[0] for call in sleep.call_args_list],
                    [0] * min(count, 4) + ([0.001] if count > 4 else []),
                )
                self.assertTrue(
                    all(
                        call.kwargs
                        == {
                            "room": 1,
                            "generation": 2,
                            "session": "s",
                            "layer": 3,
                            "slot": 0,
                            "lease_id": 4,
                        }
                        for call in query.call_args_list
                    )
                )

    def test_timeout_raises_after_query_without_followup_or_sleep(self):
        resources = SimpleNamespace(finish_chunk=unittest.mock.Mock(return_value=False))
        with (
            patch(
                "sglang.srt.disaggregation.ascend.conn.time.monotonic", return_value=10
            ),
            patch("sglang.srt.disaggregation.ascend.conn.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(TimeoutError, "scatter event did not finish"):
                _wait_for_npu_mla_bridge_chunk(
                    resources,
                    room=1,
                    generation=2,
                    session="s",
                    layer=3,
                    slot=0,
                    lease_id=4,
                    deadline=10,
                )
        self.assertEqual(resources.finish_chunk.call_count, 1)
        sleep.assert_not_called()

    def test_query_exception_propagates_without_extra_query_or_sleep(self):
        failure = RuntimeError("query failed")
        resources = SimpleNamespace(
            finish_chunk=unittest.mock.Mock(side_effect=failure)
        )
        with patch("sglang.srt.disaggregation.ascend.conn.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                _wait_for_npu_mla_bridge_chunk(
                    resources,
                    room=1,
                    generation=2,
                    session="s",
                    layer=3,
                    slot=0,
                    lease_id=4,
                    deadline=10,
                )
        self.assertEqual(resources.finish_chunk.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
