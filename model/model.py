#matthew berning - GWU, 2021
import torch
import torch.nn as nn
from torchvision import models

class Model():
    """class to hold model objects"""
    def __init__(self, model_name=None):
        """
        Keyword Argumens: 
            model_name : str, optional (default is None)
                options are currently 'vgg16' and 'resnet50' 
        """
            
        self.model_name = model_name

        if model_name == 'vgg16':
            #collect the VGG16 model architecture and weights (trained on
            #the 14million-strong ImageNet dataset) from Pytorch's Model Zoo
            #source: https://pytorch.org/vision/stable/models.html
            #paper: https://arxiv.org/abs/1409.1556
            self.model = models.vgg16(pretrained=True)
            
        elif model_name == 'resnet50':
            #collect the ResNet model architecture (pretrained on ImageNet)
            #source: https://pytorch.org/vision/stable/models.html
            #paper: https://arxiv.org/pdf/1512.03385.pdf
            self.model = models.resnet50(pretrained=True)
            
        else:
            raise Exception("No 'model_name' was specified! (try 'vgg16' or 'resnet50')")

    def construct_model(self, verbose=False):
        """
        function to replace the last fully connected
        layer of the model (1000 nodes for the
        1000 classes of ImageNet) with the custom layer
        we need to solve our problem

        Note: no activation function on the final linear
        layer since we will be using nn.CrossEntropyLoss() 
        as our loss function an that depends upon the raw
        score of the nodes being output at the end of the 
        forward pass

        Keyword Argumens: 
            verbose : bool, optional (default is False)
                control the amount of output when building
                i.e. weather to print the model
        """
        
        if self.model_name == 'vgg16':
            
            #go through the feature extraction portion of the
            #model and ensure that they will not experience weight
            #updates during training
            for param in self.model.features.parameters():
                param.requires_grad = False

            #collect the number of nodes to connect to in the
            #second to last linear layer (vgg16 it's layer 6)
            input_features = self.model.classifier[-1].in_features

            #construct our custom classification layer
            terminal_layer = nn.Linear(input_features, 2)

            #replace the original final linear layer with ours
            self.model.classifier[-1] = terminal_layer

            #check for verbosity
            if verbose: print(self.model)
                
        elif self.model_name == 'resnet50':
            
            #set all params involved in feature extraction to not have 
            #grads updated
            for param in self.model.parameters():
                param.requires_grad = False
            
            #get old last layer's in-features
            input_features = self.model.fc.in_features
            
            #make a new last layer
            terminal_layer = nn.Linear(input_features, 2)

            #replace the original final linear layer with ours
            self.model.fc = terminal_layer
            
            #this should not be needed?? Not sure what is going on ¯\_(ツ)_/¯
            for param in self.model.fc.parameters():
                param.requires_grad = True
            
            #check for verbosity
            if verbose: print(self.model)

        return self.model

