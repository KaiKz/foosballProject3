"""
Performance optimization utilities for RTX 5090 and Ryzen 9 9950X3D.
These utilities enable TF32, configure CPU threading, and standardize device handling
without changing learning algorithms or hyperparameters.
"""
import torch


def get_device():
    """
    Get the standardized device for PyTorch operations.
    Returns CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "mps")


def enable_tf32():
    """
    Enable TF32 (TensorFloat-32) for improved performance on Ampere+ GPUs.
    TF32 provides faster matrix multiplication with minimal precision loss.
    This is safe to enable and does not change algorithm behavior.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for CUDA operations")


def setup_cpu_threading(num_threads=32, num_interop_threads=8):
    """
    Configure CPU threading for optimal performance on Ryzen 9 9950X3D.
    
    Args:
        num_threads: Number of threads for intra-op parallelism (default: 32 for 16-core CPU)
        num_interop_threads: Number of threads for inter-op parallelism (default: 8)
    """
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    print(f"CPU threading configured: {num_threads} threads, {num_interop_threads} inter-op threads")


def setup_performance_optimizations(num_threads=32, num_interop_threads=8):
    """
    One-stop function to enable all performance optimizations.
    Call this at the start of training entry points.
    
    Args:
        num_threads: Number of threads for intra-op parallelism
        num_interop_threads: Number of threads for inter-op parallelism
    """
    enable_tf32()
    setup_cpu_threading(num_threads, num_interop_threads)
    device = get_device()
    print(f"Performance optimizations enabled. Using device: {device}")
    return device


def compile_model_if_supported(model, mode="max-autotune"):
    """
    Compile a PyTorch model for improved performance if supported.
    Falls back gracefully if compilation is not supported or fails.
    
    Args:
        model: PyTorch model to compile
        mode: Compilation mode (default: "max-autotune" for best performance)
    
    Returns:
        Compiled model if successful, original model otherwise
    """
    # Check if torch.compile is available (PyTorch >= 2.0)
    if not hasattr(torch, 'compile'):
        print("torch.compile not available (requires PyTorch >= 2.0), skipping compilation")
        return model
    
    try:
        compiled_model = torch.compile(model, mode=mode)
        print(f"Model compiled successfully with mode: {mode}")
        return compiled_model
    except Exception as e:
        print(f"Model compilation failed: {e}, using uncompiled model")
        return model

