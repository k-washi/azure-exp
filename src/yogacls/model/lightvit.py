from typing import Optional

from src.yogacls.LightViT.classification.lib.models.lightvit import LightViT


def load_lightvit_tiny(num_classes=1000, **kwargs):
    model_kwargs = dict(patch_size=8, embed_dims=[64, 128, 256], num_layers=[2, 6, 6],
                        num_heads=[2, 4, 8, ], mlp_ratios=[8, 4, 4], num_tokens=8,
                        num_classes=num_classes, **kwargs)
    model = LightViT(**model_kwargs)
    return model

if __name__ == "__main__":
    ckpt_path = "./pretrained/lightvit_tiny_78.7.ckpt"
    from src.yogacls.model.util import load_weight
    import torchsummary
    import torch
    
    model = load_lightvit_tiny(num_classes=2)
    model = load_weight(model, ckpt_path, head_ignore=True)
    model.eval()
    
    x = torch.rand((1, 3, 224, 224))
    y = model(x)
    print(f"RES: {y}")
    #torchsummary.summary(model, (3, 224, 224), device="cpu")