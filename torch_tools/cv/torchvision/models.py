"""
Purpose: module to 

"""

import torchvision
from torchvision import models

def models_list():
    return list(dir(torchvision.models))

def pretrained_models_list():
    return [k for k in dir(torchvision.models) if k.lower() == k]

