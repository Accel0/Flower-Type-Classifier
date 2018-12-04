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
parser = argparse.ArgumentParser(description = "Image classifier program")
parser.add_argument("data_dir" , type = str , help = "Directory used to locate source images")
parser.add_argument("--save_dir" , type = str , help = "Directory used to save checkpoints for models")
parser.add_argument("--arch" , type = str , choices = {"vgg19" , "densenet121" , "densenet201"} , help = "Type of architure to use for the NN_model")
parser.add_argument("--learning_rate" , type = float , help = "Learning rate")
parser.add_argument("--hidden_units" , type = int , help = "Number of hidden units to use for NN_model")
parser.add_argument("--epochs" , type = int , help = "Number of epochs")
parser.add_argument("--gpu" , dest = "gpu" , action = "store_true" , help = "gpu is used for training the NN_model")
parser.set_defaults(gpu = False)
arg_parse = parser.parse_args()
if arg_parse.gpu == True:
    gpu_is_on = True
    print("Gpu training is on to use")
else:
    gpu_is_on = False
    print("Cpu training is on to use")
train_loader , valid_loader , test_loader , class_to_idx = NN_model_init.create_data_dir(arg_parse.data_dir)
model , optimizer , criterion = NN_model_init.create_NN_model(arg_parse.arch , arg_parse.hidden_units , arg_parse.learning_rate)

NN_model_init.training_model(model , criterion , optimizer , arg_parse.epochs , train_loader , valid_loader , gpu_is_on)
NN_model_init.test_sets_and_validition_sets(model , test_loader , valid_loader , gpu_is_on)
NN_model_init.save_model_checkpoint(arg_parse.save_dir , model , optimizer , arg_parse.hidden_units , arg_parse.learning_rate , arg_parse.epochs , arg_parse.arch , class_to_idx)