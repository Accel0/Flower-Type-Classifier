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
def create_data_dir(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'





    data_transforms_training = transforms.Compose([transforms.RandomRotation(30) , transforms.Resize(224) , transforms.CenterCrop(224) ,    transforms.RandomHorizontalFlip() , transforms.ToTensor() , transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])])
    data_transforms_valid = transforms.Compose([transforms.Resize(256) , transforms.CenterCrop(224) , transforms.ToTensor() ,                transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])])
    data_transforms_test = transforms.Compose([transforms.Resize(256) , transforms.CenterCrop(224) , transforms.ToTensor() ,    transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])])
    image_datasets_train = datasets.ImageFolder(train_dir , transform = data_transforms_training)
    image_datasets_valid = datasets.ImageFolder(valid_dir , transform = data_transforms_valid)
    image_datasets_test = datasets.ImageFolder(test_dir , transform = data_transforms_test)
                                          
                                          
                                          
                                          
                                          
                                          
    dataloaders_train = torch.utils.data.DataLoader(image_datasets_train , batch_size = 64 , shuffle = True)
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid , batch_size = 32 , shuffle = True)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test , batch_size = 32 , shuffle = True)
    train_loader = dataloaders_train
    valid_loader = dataloaders_valid
    test_loader = dataloaders_test
    class_to_idx = image_datasets_train.class_to_idx
    return train_loader , valid_loader , test_loader , class_to_idx
def create_NN_model(arch , hidden_units , learning_rate):
    if arch == "vgg19":
        NN_model = models.vgg19(pretrained = True)
        inputs_size = NN_model.classifier[0].in_features
    elif arch == "densenet121":
        NN_model = models.densenet121(pretrained = True)
        inputs_size = NN_model.classifier.in_features
    elif arch == "densenet201":
        NN_model = models.densenet201(pretrained = True)
        inputs_size = NN_model.classifier.in_features
        
    for p in NN_model.parameters():
        p.require_grad = False
    output_size = 102
    from collections import OrderedDict as Ordered
    Linear_classifier = nn.Sequential(Ordered([("fc1" , nn.Linear(inputs_size , hidden_units)) , ("ReLU1" , nn.ReLU()) ,
                                          ("fc2" , nn.Linear(hidden_units , output_size)) , ("output" , nn.LogSoftmax(dim=1))]))
    NN_model.classifier = Linear_classifier
    optimizer = optim.Adam(NN_model.classifier.parameters() , lr = learning_rate)
    criterion = nn.NLLLoss()
    return NN_model , optimizer , criterion
def training_model(NN_model , criterion , optimizer , epochs , train_loader , valid_loader , gpu = False):
   print_every = 30
   steps = 0
   correct = 0
   total = 0
   if gpu:
        NN_model = NN_model.cuda()
   else:
        NN_model = NN_model.to("cpu")
   for e in range(epochs):
       running_loss = 0
       for ii , (inputs , labels) in enumerate(train_loader):
           steps += 1
           if gpu:
                inputs , labels = inputs.to("cuda") , labels.to("cuda")
           else:
                inputs , labels = inputs.to("cpu") , labels.to("cpu")
           optimizer.zero_grad()
           outputs = NN_model.forward(inputs)
           loss = criterion(outputs , labels)
           loss.backward()
           optimizer.step()
           running_loss += loss.item()
        
        
           if steps % print_every == 0:
               with torch.no_grad():
                   valid_loss = 0
                   for data in valid_loader:
                                       inputs , labels = data
                                       if gpu:
                                                inputs , labels = inputs.to("cuda") , labels.to("cuda")
                                       else:
                                                inputs , labels = inputs.to("cpu") , labels.to("cpu")
                                       outputs = NN_model(inputs)
                                       _, predicted = torch.max(outputs.data, 1)
                                       total += labels.size(0)
                                       correct += (predicted == labels).sum().item()
                                       valid_loss += criterion(outputs, labels).item()
               print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validiton Loss: {:.3f}.. ".format(valid_loss/print_every),
                    "Validition Accuracy: {:.3f} %".format(100 * correct / total))
               running_loss = 0
def test_sets_and_validition_sets(NN_model , test_loader , valid_loader , gpu = False):
    count = 0
    correct = 0
    total = 0
    if gpu:
        NN_model = NN_model.cuda()
    else:
        NN_model = NN_model.to("cpu")
    with torch.no_grad():
        for t in test_loader:
            inputs , labels = t
            if gpu:
                inputs , labels = inputs.cuda() , labels.cuda()
            else:
                inputs , labels = inputs.to("cpu") , labels.to("cpu")
            outputs = NN_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count += 1
            print("Test Accuracy {}:".format(count) , "{:.0f} %".format(100 * correct / total))
    print("Final testing results")
    print("Test Accuracy: {:.0f} %".format(100 * correct / total))
    count = 0
    correct = 0
    total = 0
    if gpu:
        NN_model = NN_model.cuda()
    else:
        NN_model = NN_model.to("cpu")
    with torch.no_grad():
        for v in valid_loader:
            inputs , labels = v
            if gpu:
                inputs , labels = inputs.cuda() , labels.cuda()
            else:
                inputs , labels = inputs.to("cpu") , labels.to("cpu")
            outputs = NN_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count += 1
            print("Validition Accuracy {}:".format(count) , "{:.0f} %".format(100 * correct / total))
    print("-"*10)
    print("Final testing results")
    print("Test Accuracy: {:.0f} %".format(100 * correct / total))
    print("Final validition results")
    print("validition Accuracy: {:.0f} %".format(100 * correct / total))
def save_model_checkpoint(file_location , NN_model , optimizer , hidden_units , learning_rate , epochs , arch , class_to_idx):
    NN_model.class_to_idx = class_to_idx
    NN_model.cpu()
    NN_model_save = {"arch" : arch , "optimizer" : optimizer.state_dict() , "hidden_units" : hidden_units , "learning_rate" : learning_rate , "number of epochs" : epochs , "state" : NN_model.state_dict() , "class_to_idx" : NN_model.class_to_idx}
    torch.save(NN_model_save ,  file_location)
    print("Model successfully saved!")
    
def load_model_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    NN_model , optimizer , criterion = create_NN_model(checkpoint["arch"] , checkpoint["hidden_units"] , checkpoint["learning_rate"])
    NN_model.class_to_idx = checkpoint["class_to_idx"]
    optimizer = optim.Adam(NN_model.classifier.parameters(), lr= checkpoint["learning_rate"])
    NN_model.load_state_dict(checkpoint["state"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return NN_model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    if img.size[0] > img.size[1]:
        img.thumbnail((30000 , 256))
    else:
        img.thumbnail((256 , 30000))
    bottom_side = (img.height - 224) / 2
    left_side = (img.width - 224) / 2
    up_side = bottom_side + 224
    right_side = left_side + 224
    img = img.crop((left_side , bottom_side , right_side , up_side))
    np_image = np.array(img) / 255
    mean = np.array([0.485 , 0.456 , 0.406])
    std = np.array([0.229 , 0.224 , 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2 , 0 , 1))
    return np_image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
def predict(image_path, NN_model, category_names , topk = 5 , gpu = False):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    NN_model = NN_model.cpu()
    if gpu:
        NN_model = NN_model.cuda()
    else:
        NN_model = NN_model.to("cpu")
    image_test = process_image(image_path)
    
    image_tensor = torch.from_numpy(image_test).type(torch.FloatTensor)
    
    inputs = image_tensor.unsqueeze(0)
    inputs = inputs.cpu()
    if gpu:
        inputs = inputs.to("cuda")
    else:    
        inputs = inputs.to("cpu")
    
    probs = torch.exp(NN_model.forward(inputs))
    
    topk_probs , topk_classes = probs.topk(topk)
    topk_probs = topk_probs.cpu()
    topk_probs = topk_probs.detach().numpy().tolist()[0]
    topk_classes = topk_classes.cpu()
    topk_classes = topk_classes.detach().numpy().tolist()[0]
    
    model_idx_to_class = {i : j for j , i in NN_model.class_to_idx.items()}
    top_classes = [model_idx_to_class[i] for i in topk_classes]
    top_flower_names = [cat_to_name[model_idx_to_class[i]] for i in topk_classes]
    return topk_probs , top_classes , top_flower_names

def display_image_top5(image_path , NN_model , category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    plt.figure(figsize = (6 , 10))
    ax = plt.subplot(2 , 1 , 1)
    flower_number = image_path.split("/")[2]
    flower_name = cat_to_name[flower_number]
    
    img = process_image(image_path)
    
    imshow(img , ax , title = flower_name)
    
    probs , classes , flower_names = predict(image_path , NN_model)
    
    plt.subplot(2 , 1 , 2)
    
    sns.barplot(x = probs , y = flower_names , color = sns.color_palette()[0])
    plt.show()