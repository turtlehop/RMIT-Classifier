#
# Name: Kenny Ng
# Date: 2nd August 2020
# Aim: Create CLI app to train network
#
#------------------------
#

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


#Test Line
#     python train.py 'flowers/train' --arch=vgg16 --learning_rate=0.001 --hidden_units=4096 --epochs=5 --gpu

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

import argparse
import sys
from collections import OrderedDict


def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)
        
        #images.resize_(images.shape[0], 50176)
        images.resize_(images.size()[0], 3, 224,224)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()     

    return test_loss, accuracy

def train(data_dir, arch="vgg16", lr=0.01, hidden_units=512, epochs=3, device='cpu'):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(data_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(data_dir, transform=valid_transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)        
        
    if re.search('vgg16', arch):        
        model = models.vgg16(pretrained=True)   
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            #                               ('fc2', nn.Linear(4096, 1000)),
            #                               ('relu2', nn.ReLU()),
            #                               ('dropout2', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),    
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier           

        #define Criterion and Optimizer Functions
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)          
        
    elif re.search('resnet18', arch):   
        model = models.resnet18(pretrained=True) 
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('logits', nn.Linear(hidden_units, 102))
        ]))        
        model.classifier = classifier           

        #define Criterion and Optimizer Functions
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr)   
        
    elif re.search('alexnet', arch):   
        model = models.alexnet(pretrained=True)     
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(9216, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),    
            ('output', nn.LogSoftmax(dim=1))
        ]))         
        model.classifier = classifier           

        #define Criterion and Optimizer Functions
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)           

    elif re.search('densenet', arch):   
        model = models.densenet161(pretrained=True)  
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2208, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('logits', nn.Linear(hidden_units, 102))
        ]))         
        model.classifier = classifier           

        #define Criterion and Optimizer Functions
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr)    
        
    elif re.search('inception', arch):   
        model = models.inception_v3(pretrained=True)   
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2048, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('logits', nn.Linear(hidden_units, 102))
        ]))         
        model.classifier = classifier           

        #define Criterion and Optimizer Functions
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr)                                   
     
    for param in model.parameters():
        param.requires_grad = False           




    
#     epochs = 5
    print_every = 40
    steps = 0
    running_loss = 0
#     device = 'cuda'

    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()

        model = model.to(device)

        for ii, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            steps += 1

            # Flatten images into a 50,176 long vector
            images.resize_(images.size()[0], 3, 224,224)
            #images.resize_(images.size()[0],50176)

            optimizer.zero_grad()

            output = model.forward(images)       
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)
    #                 test_loss = test_loss.to(device)
    #                 accuracy = accuracy.to(device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()    
    
    
    
    ### Save checkpoint file
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'arch': arch,
                  'learning_rate': lr,
                  'batch_size': 32,
                  'classifier' : classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
    
def main():
    parser = argparse.ArgumentParser()
#     parser.add_argument('data_dir', metavar='N', type=str, nargs='+',help='What is the location of te data directory?')
    parser.add_argument('data_dir', type=str, help='What is the location of te data directory?')
    parser.add_argument('--arch', type=str, default='vgg16', help='What is the architecture type?')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='What is the learning rate?')
    parser.add_argument('--hidden_units', type=int, default=512, help='How many units in the hidden layers?')
    parser.add_argument('--epochs', type=int, default=3, help='How many epochs?')
    parser.add_argument('--gpu', action='store_true', help='Run on GPU')
    args = parser.parse_args()

#     print(args.data_dir)
#     print(args.arch)
#     print(args.learning_rate)
#     print(args.hidden_units)   
#     print(args.epochs)
#     print(args.gpu)       
    
    if args.gpu:
        train(args.data_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, 'cuda')
    else:        
        train(args.data_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, 'cpu')

#     train(data_dir, arch="vgg16", lr=0.01, hidden_units=512, epochs=3, device='cpu')

if __name__ == '__main__':
    main()