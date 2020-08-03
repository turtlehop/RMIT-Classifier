#
# Name: Kenny Ng
# Date: 2nd August 2020
# Aim: Create CLI app to predict image
#
#------------------------
#

#Test Line
#     python predict.py "flowers/train/14/image_06050.jpg" checkpoint --top_k=5 --category_names=cat_to_name.json --gpu


import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import seaborn as sb
import json

import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import re
from PIL import Image
import numpy as np

import argparse
import sys

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    im = Image.open(image)    
    test_image_tensor = test_transforms(im)
    np_image = np.array(test_image_tensor)
    
    return torch.from_numpy(np_image)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''   
    
    image = process_image(image_path)
    image = image.to(device)
    model = model.to(device)
    
    model.eval()
    
    image.unsqueeze_(0)
    predictions = model.forward(image)   
    probabilities = torch.exp(predictions).data     
    top_ps, top_k_indices = torch.topk(probabilities,topk)    
    top_ps = np.array(top_ps.detach())[0]
    top_k_indices = np.array(top_k_indices.detach())[0]
  
    #Converting index to class
    key_list = list(model.class_to_idx.keys())
    val_list = list(model.class_to_idx.values())
    
    class_list = []
    for i in top_k_indices:        
        class_list.append(key_list[val_list.index(i)])

    return top_ps, top_k_indices

def prediction(image_path, checkpoint, top_k=5, category_names='cat_to_name.json', device='cpu'):
    
    checkpoint = checkpoint+".pth"
    model, optimizer = load_checkpoint(checkpoint)
    flower_ps, flower_index = predict(image_path, model, topk, device)

    for number in flower_index:
    #     print("index {} class {}".format(number, cat_to_name[str(number)]))
        name = cat_to_name[str(number)]
        x.append(name)    
    print(x)
    
def main():
    parser = argparse.ArgumentParser()
#     parser.add_argument('data_dir', metavar='N', type=str, nargs='+',help='What is the location of te data directory?')
    parser.add_argument('image_path', type=str, help='What is the directory path of the image?')
    parser.add_argument('checkpoint', type=str, help='Save checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='Top k results?')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='How many epochs?')
    parser.add_argument('--gpu', action='store_true', help='Run on GPU')    
    args = parser.parse_args()

    print(args.image_path)
    print(args.checkpoint)
    print(args.top_k)
    print(args.category_names)   
    print(args.gpu)   
    
    if args.gpu:
#         train(args.data_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, 'cuda')
        prediction(args.image_path, args.checkpoint, args.top_k, args.category_names, 'cuda')
    else:        
#         train(args.data_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, 'cpu') 
        prediction(args.image_path, args.checkpoint, args.top_k, args.category_names, 'cpu')
    
    
if __name__ == '__main__':
    main()    