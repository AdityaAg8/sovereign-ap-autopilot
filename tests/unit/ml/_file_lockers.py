"""The host file locker for the `gpu.lock` tests (ADR-0011 amendment 4): real `fcntl.flock` where
the host has it, else a Windows byte-range lock with the same semantics, so the flock liveness
logic in `ml.extractor_b.gpu` is exercised on every host without skips."""

from __future__ import annotations

import os
import sys
import time
from typing import Final

from ml.extractor_b.gpu import DEFAULT_LOCKER, FileLocker

# One byte far past any metadata, so readers of the lock file never hit the mandatory range lock.
_RANGE_OFFSET: Final[int] = 0x7FFF_0000
_RETRY_S: Final[float] = 0.005
_LOCK_DEADLINE_S: Final[float] = 60.0


class WindowsRangeLocker:
    """A flock stand-in for Windows test runs: `msvcrt.locking` on one byte past EOF. Like flock
    it belongs to the open handle (a second handle in the same process conflicts) and Windows
    drops it when the handle closes or the process dies. Test-only: production on Windows has no
    locker and never reclaims (ADR-0011 amendment 4)."""

    def try_lock(self, fd: int) -> bool:
        if sys.platform != "win32":
            raise RuntimeError("WindowsRangeLocker needs Windows; use the fcntl locker here")
        import msvcrt

        pos = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, _RANGE_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except PermissionError:  # EACCES: another handle holds the range
            return False
        finally:
            os.lseek(fd, pos, os.SEEK_SET)
        return True

    def lock(self, fd: int) -> None:
        # A bounded wait: a locker bug must fail the test, never hang the run.
        deadline = time.monotonic() + _LOCK_DEADLINE_S
        while not self.try_lock(fd):
            if time.monotonic() > deadline:
                raise TimeoutError(f"byte-range lock not acquired within {_LOCK_DEADLINE_S} s")
            time.sleep(_RETRY_S)


def host_locker() -> FileLocker:
    return DEFAULT_LOCKER if DEFAULT_LOCKER is not None else WindowsRangeLocker()


LOCKER: Final[FileLocker] = host_locker()
