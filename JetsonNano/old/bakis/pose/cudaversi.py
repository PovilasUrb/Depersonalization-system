import torch
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available?", torch.cuda.is_available())
print("Device properties:", torch.cuda.get_device_properties(0))
