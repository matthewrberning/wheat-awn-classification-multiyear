#matthew berning
import os
import sys
import time
import argparse

from model.model import Model
from model.dataset import WheatAwnDataset

import fire
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

def tensor_to_image(tensor):
    '''helper function to take a tensor and image-ize it'''
    
    #take the tensor representation of the image and numpyify it
    image = tensor.clone().detach().cpu().numpy()
    
    #re order the dimensionality for matplotlib
    image = image.transpose(1, 2, 0)
    
    image = image.clip(0,1)
    
    return image


def add_subplot_border(ax, width=1, color=None ):
    '''helper function to add a border to the plot'''

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
    fig = plt.figure(figsize=(50., 30.),facecolor="w")
    
    awned_colormap = plt.get_cmap('YlOrRd')
    awnless_colormap = plt.get_cmap('YlGnBu')

    grid = ImageGrid(fig, 
                     rect=111,  # similar to subplot(111)
                     nrows_ncols=(10, 5),  # creates 2x2 grid of axes
                     axes_pad=(0.3, 0.7),  # pad between axes in inch. (horizontal padding, vertical padding)
                     )

    for ax, im, pred, grndth, plot_id, conf in zip(grid, imgs, predicted_labels, groundtruth_labels, plot_id_GT, pred_confs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        
        add_subplot_border(ax, width=6.5, color=awned_colormap(conf) if pred==0 else awnless_colormap(conf))
        
        plot = list(plot_id_dict.keys())[list(plot_id_dict.values()).index(plot_id.item())]
        
        ax.set_axis_off()
        
        ax.text(0.0,1.11, f"Pr: {pred} (GT: {grndth})  {round(conf*100,2)}%\nplot_id: {plot}", transform=ax.transAxes)


    
    fig.text(x=0.505, y=0.915, s=fig_title, fontsize=28, ha="center", transform=fig.transFigure)
    fig.text(x=0.5, y=0.90, s=f"model: {model_pth_name}", fontsize=18, ha="center", transform=fig.transFigure)

    #output
    
    fig.savefig(os.path.join(save_dir, f"{save_prefix}_{count_saves}.png"))
    plt.close()


def model_prediction_montages(data_csv,
                              batch_size, 
                              fig_title, 
                              model_pth,
                              pkl_file_path, 
                              end_after=None, 
                              find_incorrects=True, 
                              dataset_path='/pless_nfs/home/matthewrberning/multi-year-cult-class/data/preprocessed/',
                              save_dir='/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/data/montages'):

    print(f"loading model from: {model_pth}")
    saved_model = Model().construct_model(verbose=False)
    saved_model.load_state_dict(torch.load(model_pth))
    
    #send the model to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saved_model = saved_model.to(device)

    #collect test dataset and create loader iterable-object
    transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])
    data = WheatAwnDataset(csv_filepath=data_csv, dataset_dir=dataset_path, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    #collect the dictionary that tells us which plot id number cooresponds to which plot_id string
    with open(pkl_file_path, 'rb') as handle:
        plot_id_dict = pickle.load(handle)

    #remove model pth file name to add to figure
    model_pth_name = model_pth.split('/')[-1] 

    #create the prefix to be added to each saved montage
    save_prefix = f"{model_pth_name.split('.')[0]}_{'incorrects' if (find_incorrects==True) else 'corrects'}_{data_csv.split('.')[0]}"

    print("\n\ndata csv: ", data_csv)
    print("batch_size: ", batch_size)
    print("fig_title: ", fig_title)
    print("model_pth file: ", model_pth_name)
    print("plot id pkl file: ", pkl_file_path)
    print("save_prefix: ", save_prefix)
    print("end_after: ", end_after)
    print("find_incorrects: ", find_incorrects,"\n\n")

    #set model mode
    saved_model.eval()

    imgs = []
    predicted_labels = []
    groundtruth_labels = []
    plot_id_GT = []
    pred_confs = []

    b = 0 #breaker
    bb = 0 #bbreaker

    count_saves = 1

    #make sure to not accumulate gradients
    with torch.no_grad():

        progress_bar = tqdm(dataloader, total=int(len(dataloader)), desc=f"{fig_title} Progress: ")

        for step, data in enumerate(progress_bar):

            images, labels, plot_ids = data[0], data[1], data[2]
            
            if end_after:
                if bb==end_after:
                    print(f"breaking after {bb}")
                    break

            #send the tensors to the device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            #images = images.float() #uncomment if using read_image() from torch
            outputs = saved_model(images)

            #find the predicted classes indicies
            _, preds = torch.max(outputs, 1)

            soft_preds = torch.softmax(outputs, 1)

            #print("preds: ", list(preds.cpu().numpy()))
            #print("labels: ", list(labels.cpu().numpy()))


            for index, prediction in enumerate(preds):
                if find_incorrects:
                    if prediction == labels[index]:
                        continue
                        #print("correct!")

                    else:
                        #print("false!")
                        imgs.append(tensor_to_image(images[index]))
                        predicted_labels.append(prediction.detach().cpu().numpy())
                        groundtruth_labels.append(labels[index].detach().cpu().numpy())
                        plot_id_GT.append(plot_ids[index].detach().cpu().numpy())
                        pred_confs.append(soft_preds[index].cpu().numpy()[prediction])
                        b+=1 #add to breaker when we've accumulated another false preditcion

                        if b==50:
                            #we've collected 50 (b == 50) to plot
                            print(f"resetting after {b}")
                            save_plot_incorrects_grid(imgs, 
                                                      predicted_labels, 
                                                      groundtruth_labels, 
                                                      plot_id_GT, 
                                                      fig_title, 
                                                      model_pth_name, 
                                                      plot_id_dict, 
                                                      pred_confs, 
                                                      save_prefix, 
                                                      count_saves, 
                                                      save_dir)

                            print("plot saved, resetting")
                            count_saves+=1
                            bb+=b
                            b = 0
                            imgs = []
                            predicted_label = []
                            groundtruth_label = []
                else:
                    if prediction == labels[index]:
                        #print("correct!")
                        imgs.append(tensor_to_image(images[index]))
                        predicted_labels.append(prediction.detach().cpu().numpy())
                        groundtruth_labels.append(labels[index].detach().cpu().numpy())
                        plot_id_GT.append(plot_ids[index].detach().cpu().numpy())
                        pred_confs.append(soft_preds[index].cpu().numpy()[prediction])
                        b+=1 #add to breaker when we've accumulated another CORRECT preditction

                        if b==50:
                            #we've collected 50 (b == 50) to plot
                            print(f"resetting after {b}")
                            save_plot_incorrects_grid(imgs, 
                                                      predicted_labels, 
                                                      groundtruth_labels, 
                                                      plot_id_GT, 
                                                      fig_title, 
                                                      model_pth_name, 
                                                      plot_id_dict, 
                                                      pred_confs, 
                                                      save_prefix, 
                                                      count_saves, 
                                                      save_dir)

                            print("plot saved, resetting")
                            count_saves+=1
                            bb+=b
                            b = 0
                            imgs = []
                            predicted_label = []
                            groundtruth_label = []

                    else:
                        #print("false!")
                        continue







if __name__ == '__main__':

    #find the correct GPU -and use it!
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.3, maxMemory = 0.3, includeNan=False, excludeID=[], excludeUUID=[])

    print("GPU Chosen: ", str(deviceIDs[0]))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])

    fire.Fire(model_prediction_montages)

