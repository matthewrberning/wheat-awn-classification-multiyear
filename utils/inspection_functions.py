#matthew berning - GWU, 2021

import os
import sys
import time
import argparse
from os.path import dirname, abspath

sys.path.insert(0, os.path.abspath(os.path.dirname('../model/')))
sys.path.insert(0, os.path.abspath(os.path.dirname('../data/')))

from model import Model
from dataset import WheatAwnDataset
from input import yesno
from tensor_operations import tensor_to_image

import PIL
import pickle
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

def add_subplot_border(ax, width=1, color=None ):
    '''
    helper function to add a border to the plot

    input: ax - (mpl ax object) the subplot to be added to

    input: width - (number, default: 1) the width of the border

    input: color - (string, default: None) the rgba, hex, or named matplotlib color for the border
    '''

    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        color=color,
        transform=ax.transAxes,
        zorder=-1,
        lw=2*width+1,
        fill=None,
    )
    fig.patches.append(rect)


def save_plot_incorrects_grid(imgs, 
                              predicted_labels, 
                              groundtruth_labels, 
                              plot_id_GT, 
                              fig_title, 
                              model_pth_name, 
                              plot_id_dict, 
                              pred_confs, 
                              save_prefix, 
                              count_saves,
                              save_dir):

    '''helper function to plot the gird of images'''

    #make the figure object and set the background color to white
    fig = plt.figure(figsize=(50., 30.),facecolor="w")
    
    #make color maps for the two classes (awned==0 awnless==1)
    awned_colormap = plt.get_cmap('YlOrRd')
    awnless_colormap = plt.get_cmap('YlGnBu')

    grid = ImageGrid(fig, 
                     rect=111,  # similar to subplot(111)
                     nrows_ncols=(10, 5),  # creates 2x2 grid of axes
                     axes_pad=(0.3, 0.7),  # pad between axes in inch. (horizontal padding, vertical padding)
                     )

    #zip together the axes from the grid, the images, the predictions, ground truths
    #plot_id's, and prediction confidences
    for ax, im, pred, grndth, plot_id, conf in zip(grid, imgs, predicted_labels, groundtruth_labels, plot_id_GT, pred_confs):
        
        #add the image
        ax.imshow(im)
        
        #add the boarder (intensity/color wrt. to the confidence and class predicted)
        add_subplot_border(ax, width=6.5, color=awned_colormap(conf) if pred==0 else awnless_colormap(conf))
        
        #find the correct plot_id string for the number representation
        plot = list(plot_id_dict.keys())[list(plot_id_dict.values()).index(plot_id.item())]
        
        ax.set_axis_off()
        
        #add a text element for the caption
        ax.text(0.0,1.11, f"Pr: {pred} (GT: {grndth})  {round(conf*100,2)}%\nplot_id: {plot}", transform=ax.transAxes)


    #add the title and model pth name to the figure as text elements
    fig.text(x=0.505, y=0.915, s=fig_title, fontsize=28, ha="center", transform=fig.transFigure)
    fig.text(x=0.5, y=0.90, s=f"model: {model_pth_name}", fontsize=18, ha="center", transform=fig.transFigure)

    #save off the figure to the disk    
    fig.savefig(os.path.join(save_dir, f"{save_prefix}_{count_saves}.png"))
    plt.close()

