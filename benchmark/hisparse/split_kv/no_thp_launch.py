"""Disable transparent huge pages for this server process and its children."""

import ctypes
import runpy
import sys


def main():
    """Disable THP, then run the standard SGLang server entry point."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE failed")
    sys.argv[0] = "sglang.launch_server"
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
