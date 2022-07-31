import json
import pickle
from typing import Literal

import torch

def save_as_pickle(obj, save_path: str):
    with open(save_path, mode='wb') as f:
        pickle.dump(obj, f)
    print(f'Save to {save_path}.')

def save_dict(obj: dict, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(obj, f, indent=4)
    print(f'Save to {save_path}.')


def save_model(model, save_path: str):
    torch.save(model.state_dict(), save_path)
    print(f'Save to {save_path}.')


Device = Literal['cpu', 'cuda']

"""
refs: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-on-gpu-load-on-gpu
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
"""
def load_model(model, load_path: str, _device: Device):
    device = torch.device(_device)
    model.load_state_dict(torch.load(load_path))
    model.to(device)
    return model

