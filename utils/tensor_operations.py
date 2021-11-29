#matthew berning - GWU, 2021
import os
import sys
import torch

def tensor_to_image(tensor):
    '''
    helper function to take a tensor and image-ize it

    input: tensor - (torch tensor) a tensorized image

    output: image - (numpy array) the image as a clipped/transposed numpy array
    '''
    
    #take the tensor representation of the image and numpyify it
    image = tensor.clone().detach().cpu().numpy()
    
    #re order the dimensionality for matplotlib
    image = image.transpose(1, 2, 0)
    
    image = image.clip(0,1)
    
    return image