import sys
import os
import shutil
import ctypes
import torch
import torchvision

def inspect_environment():
    print("=== SYSTEM & PYTORCH ENVIRONMENT AUDIT ===")
    print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Total VRAM: {props.total_memory / (1024**3):.2f} GB")
        print(f"CUDA Capability: {props.major}.{props.minor}")
    else:
        print("GPU Name: None (CPU mode only)")
        print("VRAM: N/A")
        
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    print(f"System Total RAM: {mem.ullTotalPhys / (1024**3):.2f} GB")
    print(f"System Available RAM: {mem.ullAvailPhys / (1024**3):.2f} GB")
    
    c_usage = shutil.disk_usage("C:/")
    print(f"Disk C: Total={c_usage.total / (1024**3):.2f} GB, Free={c_usage.free / (1024**3):.2f} GB")
    
    if os.path.exists("D:/"):
        d_usage = shutil.disk_usage("D:/")
        print(f"Disk D: Total={d_usage.total / (1024**3):.2f} GB, Free={d_usage.free / (1024**3):.2f} GB")

if __name__ == "__main__":
    inspect_environment()
