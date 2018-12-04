import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
import seaborn as sns
import json
import time
from PIL import Image
import NN_model_init

import argparse

parser = argparse.ArgumentParser(description = "Image prediction program")
parser.add_argument("input" , type = str , help = "Path to image")
parser.add_argument("checkpoint" , type = str , help = "Path to model to use for prediction")
parser.add_argument("--topk" , type = int , help = "Number of top images to show")
parser.add_argument("--category_names" , type = str , help = "Category names to switch to real world names")
parser.add_argument("--gpu" , dest = "gpu" , action = "store_true" , help = "gpu is used for training the NN_model")
parser.set_defaults(gpu = False)
arg_parse = parser.parse_args()
if arg_parse.gpu:
    gpu_is_on = True
    print("Gpu is on to use for prediction")
else:
    print("Cpu is on to use for prediction")




NN_model = NN_model_init.load_model_checkpoint(arg_parse.checkpoint)
print(NN_model_init.predict(arg_parse.input, NN_model, arg_parse.category_names , arg_parse.topk , gpu_is_on))