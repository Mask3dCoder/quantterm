"""CLI utilities."""
import sys


class SuppressStderr:
    """Context manager to suppress stderr output."""
    def __enter__(self):
        self._stderr = sys.stderr
        devnull = 'NUL' if sys.platform == 'win32' else '/dev/null'
        sys.stderr = open(devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self._stderr
