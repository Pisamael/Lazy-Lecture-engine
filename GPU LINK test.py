import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
