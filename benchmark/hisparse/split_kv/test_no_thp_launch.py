"""CPU-only tests for the THP server launcher wrapper."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

WRAPPER_PATH = Path(__file__).with_name("no_thp_launch.py")


def load_wrapper(name):
    spec = importlib.util.spec_from_file_location(name, WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNoThpLaunch(unittest.TestCase):
    def test_import_is_safe_for_normal_and_spawn_module_names(self):
        for name in ("no_thp_launch_import", "__mp_main__"):
            with self.subTest(name=name):
                argv = ["wrapper.py", "--model", "example"]
                with (
                    patch.object(sys, "argv", argv.copy()),
                    patch("ctypes.CDLL") as load_libc,
                    patch("runpy.run_module") as run_server,
                ):
                    original_argv = sys.argv.copy()
                    load_wrapper(name)
                    self.assertEqual(sys.argv, original_argv)
                    load_libc.assert_not_called()
                    run_server.assert_not_called()

    def test_main_disables_thp_and_runs_server_once_with_forwarded_arguments(self):
        module = load_wrapper("no_thp_launch_main")
        libc = Mock()
        libc.prctl.return_value = 0
        argv = ["wrapper.py", "--model", "example", "--port", "30000"]
        observed_argv = []
        with (
            patch.object(sys, "argv", argv.copy()),
            patch.object(module.ctypes, "CDLL", return_value=libc) as load_libc,
            patch.object(
                module.runpy,
                "run_module",
                side_effect=lambda *args, **kwargs: observed_argv.append(
                    sys.argv.copy()
                ),
            ) as run_server,
        ):
            module.main()
            self.assertEqual(
                sys.argv,
                ["sglang.launch_server", "--model", "example", "--port", "30000"],
            )

        load_libc.assert_called_once_with(None, use_errno=True)
        libc.prctl.assert_called_once_with(41, 1, 0, 0, 0)
        run_server.assert_called_once_with("sglang.launch_server", run_name="__main__")
        self.assertEqual(len(observed_argv), 1)
        self.assertEqual(
            observed_argv[0],
            ["sglang.launch_server", "--model", "example", "--port", "30000"],
        )

    def test_prctl_failure_raises_oserror_without_starting_server(self):
        module = load_wrapper("no_thp_launch_failure")
        libc = Mock()
        libc.prctl.return_value = -1
        with (
            patch.object(sys, "argv", ["wrapper.py", "--model", "example"]),
            patch.object(module.ctypes, "CDLL", return_value=libc),
            patch.object(module.ctypes, "get_errno", return_value=13),
            patch.object(module.runpy, "run_module") as run_server,
        ):
            with self.assertRaises(OSError) as raised:
                module.main()

        self.assertEqual(raised.exception.errno, 13)
        self.assertIn("PR_SET_THP_DISABLE failed", str(raised.exception))
        run_server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
