import torch
import torch.nn as nn

pad = nn.ReflectionPad2d(1)
input_image = torch.tensor([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]], dtype=torch.float32)

output_image = pad(input_image)

print(output_image)