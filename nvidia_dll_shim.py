"""Windows CUDA-DLL shim for faster-whisper / ctranslate2.

The pip `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels drop their DLLs in
`<venv>/Lib/site-packages/nvidia/*/bin`, which is NOT on the Windows DLL search
path. Worse, ctranslate2 loads `cublas64_12.dll` without first resolving its
`cublasLt64_12.dll` dependency, so `os.add_dll_directory` alone isn't enough and
you get: RuntimeError('Library cublas64_12.dll is not found or cannot be loaded').

Fix: add every nvidia */bin dir to the DLL search path AND to PATH, then
pre-load the CUDA libs in dependency order so ctranslate2's later lazy loads
find them already resident.

Usage — call this BEFORE constructing a GPU WhisperModel:

    from nvidia_dll_shim import enable
    enable()
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
"""
import os
import sys
import ctypes

# dependency order matters: cublasLt before cublas; cudnn sublibs before cudnn64_9
_PRELOAD = (
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn64_9.dll",
)


def enable(verbose=False):
    """Make the pip nvidia CUDA wheels loadable. Returns the list of bin dirs."""
    base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if not os.path.isdir(base):
        if verbose:
            print("nvidia_dll_shim: no nvidia wheels found at", base)
        return []
    bins = []
    for sub in sorted(os.listdir(base)):
        d = os.path.join(base, sub, "bin")
        if os.path.isdir(d):
            bins.append(d)
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    for name in _PRELOAD:
        for d in bins:
            p = os.path.join(d, name)
            if os.path.exists(p):
                try:
                    ctypes.WinDLL(p)
                    if verbose:
                        print("nvidia_dll_shim: preloaded", name)
                except OSError as e:
                    if verbose:
                        print("nvidia_dll_shim: skip", name, "->", e)
                break
    return bins


if __name__ == "__main__":
    dirs = enable(verbose=True)
    print("nvidia_dll_shim: enabled", len(dirs), "dirs")
