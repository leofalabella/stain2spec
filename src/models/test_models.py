import firstmodel as m
import torch
gen = m.GeneratorUNet()
dummy_input = torch.randn(1,3,256,256)
out = gen(dummy_input)
print(out.shape)