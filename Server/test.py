import torch

if torch.xpu.is_available():
    x = torch.tensor([1.0, 2.0, 3.0], device="xpu")
    print("Tensor on XPU:", x)
    print("x * 2 =", x * 2)
else:
    print("XPU not available!")
