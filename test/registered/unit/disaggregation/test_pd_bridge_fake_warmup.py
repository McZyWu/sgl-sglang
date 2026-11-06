import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestBridgeFakeWarmup(CustomTestCase):
    """Fake startup requests must not acquire a real MLA bridge room."""

    def _fixture(self, receiver, req):
        from types import SimpleNamespace
        from unittest.mock import Mock

        req.time_stats = SimpleNamespace(set_decode_transfer_queue_entry_time=Mock())
        item = SimpleNamespace(req=req, kv_receiver=receiver, is_rebootstrap=False)
        queue = SimpleNamespace(
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(alloc=lambda: 7),
            _transfer_page_indices=lambda *_: [3, 4],
            draft_token_to_kv_pool=None,
            transfer_queue=SimpleNamespace(enable_staging=False),
            _num_published_destinations=0,
        )
        return queue, item

    def test_fake_warmup_publishes_metadata_without_bridge_room(self):
        from types import SimpleNamespace

        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.decode import DecodePreallocQueue
        from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        manager = SimpleNamespace(npu_mla_bridge_metadata=object())
        receiver = FakeKVReceiver(manager, "unused", 0)
        receiver.init(0)
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        queue, item = self._fixture(
            receiver, SimpleNamespace(bootstrap_host=FAKE_BOOTSTRAP_HOST)
        )
        DecodePreallocQueue._send_kv_metadata(queue, item, [1, 2], 128)
        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertEqual(queue._num_published_destinations, 1)
        self.assertEqual(item.metadata_buffer_index, 7)
        item.req.time_stats.set_decode_transfer_queue_entry_time.assert_called_once()

    def test_real_bridge_prepares_before_sending_metadata(self):
        from types import SimpleNamespace

        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        events = []
        receiver = SimpleNamespace(
            kv_mgr=SimpleNamespace(npu_mla_bridge_metadata=object()),
            prepare_npu_mla_bridge_room=lambda *args: events.append(("prepare", args)),
            send_metadata=lambda *args, **kwargs: events.append(("send", args)),
        )
        queue, item = self._fixture(
            receiver, SimpleNamespace(bootstrap_host="192.0.2.1")
        )
        DecodePreallocQueue._send_kv_metadata(queue, item, [1, 2], 128)
        self.assertEqual([event[0] for event in events], ["prepare", "send"])
        self.assertEqual(events[0][1][1:4], ([3, 4], 2, 0))
        self.assertEqual(events[1][1][:2], ([3, 4], 7))
        self.assertEqual(queue._num_published_destinations, 1)

    def test_real_bridge_missing_prepare_fails_before_send(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        receiver = SimpleNamespace(
            kv_mgr=SimpleNamespace(npu_mla_bridge_metadata=object()),
            send_metadata=Mock(),
        )
        queue, item = self._fixture(
            receiver, SimpleNamespace(bootstrap_host="192.0.2.1")
        )
        with self.assertRaises(AttributeError):
            DecodePreallocQueue._send_kv_metadata(queue, item, [1, 2], 128)
        receiver.send_metadata.assert_not_called()
        self.assertEqual(queue._num_published_destinations, 0)

    def test_disabled_bridge_does_not_require_fake_request_fields(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        receiver = SimpleNamespace(kv_mgr=SimpleNamespace(), send_metadata=Mock())
        queue, item = self._fixture(receiver, SimpleNamespace())
        DecodePreallocQueue._send_kv_metadata(queue, item, [1, 2], 128)
        receiver.send_metadata.assert_called_once_with([3, 4], 7, None)
        self.assertEqual(queue._num_published_destinations, 1)


if __name__ == "__main__":
    unittest.main()
