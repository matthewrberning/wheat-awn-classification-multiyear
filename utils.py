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


def yesno(question):
    """
    Simple Yes/No Function.

    input: question - (string) the question to be responded to with (y/n)

    returns True if y/Y
    returns False if n/N

    (otherwise recurses for valid answer)
    from
    """
    prompt = f'{question} (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    return False


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


def build_conditional_montages(data_csv,
                              batch_size, 
                              fig_title, 
                              model_pth,
                              pkl_file_path, 
                              end_after=None, 
                              find_incorrects=True, 
                              dataset_path='./data/preprocessed/',
                              save_dir='./data/montages',
                              verbose=False):
    """
    function to make conditinal montages (i.e. montages of correctly classified images
    or montages of incorrectly classified images) of 50 images each that include the
    confidence in the prediction, the prediction itself, and the plot_id (translated from 
    a static numeric dctionary). The montages are saved to the save_dir location.

    Keyword Argumens: 
        data_csv : string, required 
            the .CSV file containing the dataset description (labels, images, plot_ids)
            that can be interpreted by the dataloader/model

        batch_size : int, required 
            the number of images to be returned each iteration of the dataloader

        fig_title : string, required 
            the text to appear at the top of the figure (will also be used for the progress bar)
            e.g. "Incorrect Predictions - (2020 test set) [1==awnless]"

        model_pth : string, required 
            the path to the model pth file to be used to make the predictions

        pkl_file_path : string, required 
            the path to the pickled dictinary that contains the relationship between the 
            numeric representation of the plot_id and the string value

        end_after : int, optional (default is None)
            a limiter on the number of images fitting a specific contition (correct/incorrect)
            to find/make into montages -ideally a multiple of 50
            e.g. end_after=2000 will make 40 montages of 50 images

        find_incorrects : bool optional (default is True)
            the switch to flip for locating (montaging) correct preditctions vs. incorrect ones.
            usually we're interested in the incorrect predictions so that's the default

        dataset_path : str, optional (default is './data/preprocessed')
            path to the location where the images are saved on disk, needed for the dataloader

        save_dir : string, optional (default is './data/montages')
            location to save the finished montages

        verbose : bool, optional (default is False)
            control amount of output regarding saving, etc.
            
    """
    print("\n\nbuilding montages...")

    #load up the model
    print(f"\nloading model from: {model_pth}")
    saved_model = Model().construct_model(verbose=False)
    saved_model.load_state_dict(torch.load(model_pth))
    
    #send the model to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nsending model to device: ", device)
    saved_model = saved_model.to(device)

    #collect test dataset and create loader iterable-object
    transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])
    data = WheatAwnDataset(csv_filepath=data_csv, dataset_dir=dataset_path, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    #collect the dictionary that tells us which plot id number cooresponds to which plot_id string
    with open(pkl_file_path, 'rb') as handle:
        plot_id_dict = pickle.load(handle)

    #remove model pth file name as a seperate string (so we can embedd it in the final figures)
    model_pth_name = model_pth.split('/')[-1] 

    #create the prefix to be added to each saved montage
    save_prefix = f"{model_pth_name.split('.')[0]}_{'incorrects' if (find_incorrects==True) else 'corrects'}_in_{data_csv.split('/')[-1].split('.')[0]}_dataset"

    #output to 
    print("\n\ndata csv: ", data_csv)
    print("batch_size: ", batch_size)
    print("fig_title: ", fig_title)
    print("model_pth file: ", model_pth_name)
    print("plot id pkl file: ", pkl_file_path)
    print("save_prefix: ", save_prefix)
    print("end_after: ", end_after)
    print("find_incorrects: ", find_incorrects,"\n\n")

    #validate format
    if yesno("is the above format correct?"):

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
                                if verbose: print(f"resetting after {b}")
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

                                if verbose: print("plot saved, resetting")
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
                                if verbose: print(f"resetting after {b}")
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

                                if verbose: print("plot saved, resetting")
                                count_saves+=1
                                bb+=b
                                b = 0
                                imgs = []
                                predicted_label = []
                                groundtruth_label = []

                        else:
                            #print("false!")
                            continue
    else:
        print("\n\n..need to reconfigure!")
        return






if __name__ == '__main__':

    #find the correct GPU -and use it!
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.3, maxMemory = 0.3, includeNan=False, excludeID=[], excludeUUID=[])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])

    print("GPU Chosen: ", str(deviceIDs[0]))

    #use the fire module to make a CLI out of the montage function
    fire.Fire(build_conditional_montages)

