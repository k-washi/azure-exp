from pyparsing import Optional
import torch
import os

STATE_DICT_KEY = "state_dict"
MODEL_HEAD_KEY = "head"

def load_ckpt(ckpt_path:str):
    _ckpt = torch.load(ckpt_path)
    if isinstance(_ckpt, dict) and STATE_DICT_KEY in _ckpt:
        _ckpt = _ckpt[STATE_DICT_KEY]
    return _ckpt

def load_weight(model:torch.nn.Module, ckpt_path:str, head_ignore=False):
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} is not found.")
    ckpt = load_ckpt(ckpt_path)
    model_dict = model.state_dict()
    
    pretrained_dict = {}
    for k, v in ckpt.items():
        if k in model_dict:
            if head_ignore and MODEL_HEAD_KEY in k:
                # headは無視
                continue
            pretrained_dict[k] = v
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model
    

if __name__ == "__main__":
    ckpt_path = "./pretrained/lightvit_tiny_78.7.ckpt"
    load_ckpt(ckpt_path)