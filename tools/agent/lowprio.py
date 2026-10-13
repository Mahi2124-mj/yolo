"""import lowprio  -> this process runs at below-normal priority (the live monitor keeps the CPU).
The handle types must be declared: without them ctypes passes the 64-bit pseudo
handle (-1) as a 32-bit int and the call fails silently."""
import ctypes
from ctypes import wintypes

try:
    _k = ctypes.WinDLL("kernel32", use_last_error=True)
    _k.GetCurrentProcess.restype = wintypes.HANDLE
    _k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k.SetPriorityClass.restype = wintypes.BOOL
    OK = bool(_k.SetPriorityClass(_k.GetCurrentProcess(), 0x00004000))
except Exception:
    OK = False
