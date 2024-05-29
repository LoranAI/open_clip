import torch
import sys
import os

# Add the src directory to the sys.path
sys.path.append(os.path.abspath('/home/lbaiardi/open_clip/src'))

# Now you can import the model module
from open_clip import model

batch_size = 2
image_size = 224
context_length = 77
images = torch.randn(batch_size, 3, image_size, image_size)
texts = torch.randint(0, 49408, (batch_size, context_length))

embed_dim = 728
vision_cfg = model.SimplexCLIPVisionCfg()
text_cfg = model.SimplexCLIPTextCfg()
model = model.PolytopeCLIP(embed_dim=embed_dim, vision_cfg=vision_cfg, text_cfg=text_cfg)
    
output = model(image=images, text=texts)

print(isinstance(output, tuple))
print(len(output) == 3)  # image_features, text_features, logit_scale
print(output[0] is not None)  # image_features
print(output[1] is not None)  # text_features
print(output[2] is not None)  # logit_scale
print(output[0].shape)
print(output[1].shape)
print(output[2].shape)
