import torch
import sys
import os

# Add the src directory to the sys.path
sys.path.append(os.path.abspath('/home/dtanasa/open_clip/src'))

# Now you can import the model module
from open_clip import model


batch_size = 2
image_size = 224
context_length = 77
images = torch.randn(batch_size, 3, image_size, image_size)
texts = torch.randint(0, 49408, (batch_size, context_length))

embed_dim = 512
vision_cfg = model.SimplexCLIPVisionCfg()
text_cfg = model.SimplexCLIPTextCfg()

model = model.PolytopeCLIP(embed_dim=embed_dim, vision_cfg=vision_cfg, text_cfg=text_cfg)
    
output = model(image=images, text=texts)

assert isinstance(output, tuple)
assert len(output) == 3  # image_features, text_features, logit_scale
assert output[0] is not None  # image_features
assert output[1] is not None  # text_features
assert output[2] is not None  # logit_scale
