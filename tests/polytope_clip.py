import torch
import sys
import os
import unittest
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

# Add the src directory to the sys.path
sys.path.append(os.path.abspath('/home/lbaiardi/open_clip/src'))

# Now you can import the model module
from open_clip import model
from open_clip.openai import load_openai_polytope_model
from open_clip.model import CLIPTextCfg


"""
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
"""

# Test the model
class PolytopeCLIPTest(unittest.TestCase):

    def test_output(self):
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

        self.assertTrue(isinstance(output, tuple))
        self.assertTrue(len(output) == 3)
        self.assertTrue(output[0] is not None)
        self.assertTrue(output[1] is not None)
        self.assertTrue(output[2] is not None)
        self.assertEqual(output[0].shape, torch.Size([batch_size, embed_dim]))
        self.assertEqual(output[1].shape, torch.Size([batch_size, embed_dim]))
        self.assertEqual(output[2].shape, torch.Size([]))

    def test_pretrained_polytope_clip_from_clip(self):
        model_name = "ViT-B/32"
        model_name = model_name.replace("/", "-")
        precision = 'fp32' if torch.cuda.is_available() else 'fp16'
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_openai_polytope_model(model_name, precision=precision, device=device)

        self.assertTrue(isinstance(model, model.PolytopeCLIP))


if __name__ == '__main__':
    unittest.main()