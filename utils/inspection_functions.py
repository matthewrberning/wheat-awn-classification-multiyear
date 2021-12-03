#matthew berning - GWU, 2021

import os
import sys
import time
import argparse
from os.path import dirname, abspath

super_dir = dirname(abspath(__file__))
sys.path.insert(0, super_dir)
sys.path.insert(0, os.path.join(super_dir, '../model/'))
sys.path.insert(0, os.path.join(super_dir, '../data/'))
# sys.path.insert(0, os.path.abspath(os.path.dirname('../model/')))
# sys.path.insert(0, os.path.abspath(os.path.dirname('../data/')))


from model import Model
from dataset import WheatAwnDataset
from input_validation import yesno
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

def get_montages(model_name):
    search_dir = "/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/data/montages"
    os.chdir(search_dir)
    mont_files = filter(os.path.isfile, os.listdir(search_dir))
    mont_files = [os.path.join(search_dir, f) for f in mont_files]# add path to each file

    mont_files = [file for file in mont_files if file.endswith('.png')] #only get image (montage) files

    corrects = []
    incorrects = []
    
    for file in mont_files:
        if model_name in file:
            if 'incorrects' in file:
                incorrects.append(file)
            elif '_corrects' in file:
                corrects.append(file)
    
    print(f"found {len(corrects)} 'corrects' montages")
    print(f"found {len(incorrects)} 'incorrects' montages")
    
    return corrects, incorrects

def get_confusion_matrix_for_dataset(data_csv, batch_size, saved_model, title, device):
    '''
    helper function to wrap up generating a dataloader, collecting
    predictions, and plotting a confusion matrix for a specific dataset
    '''
    
    #collect test dataset and create loader iterable-object
    dataset_path = '/pless_nfs/home/matthewrberning/multi-year-cult-class/data/preprocessed/'
    
    transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])

    data = WheatAwnDataset(csv_filepath=data_csv, dataset_dir=dataset_path, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    #track the correct predictions
    corrects = 0.0

    #set model mode
    saved_model.eval()

    # Get the predictions/GT's for the confusion matrix
    predictions = []
    ground_truths = []

    #make sure to not accumulate gradients
    with torch.no_grad():

        progress_bar = tqdm(dataloader, total=int(len(dataloader)), desc=f"{title} Progress: ")

        for step, data in enumerate(progress_bar):

            #unpack the data from the progress bar
            images, labels = data[0], data[1]

            #send the tensors to the device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            #images = images.float() #uncomment if using read_image() from torch
            outputs = saved_model(images)

            #find the predicted classes indicies
            _, preds = torch.max(outputs, 1)

            #track the correct predictions
            corrects += torch.sum(preds == labels.data).item()

            ground_truths.extend(list(labels.cpu().numpy()))
            predictions.extend(list(preds.cpu().numpy()))
            
    #calculate the accuracy 
    accuracy = (corrects/len(dataloader.dataset))*100
    accuracy = "{:5.2f}".format(accuracy)
    print("accuracy: ", accuracy)
    print('\n')
    print("predictions:   ", len(predictions))
    print("ground truths: ", len(ground_truths))
    
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true=ground_truths, y_pred=predictions)
    
    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(title, fontsize=18)
    plt.show()

def plot_training_history(history_dict_pkl_path):
    '''helper function to take a history dictionary pickel file and plot it
    
    history dict structure:
    
                 h_d = {'date': current_time,
                        'epochs': epoch+1,
                        'training_loss_history':training_loss_history,
                        'training_accuracy_history':training_accuracy_history,
                        'validation_loss_history':validation_loss_history,
                        'validation_accuracy_history':validation_accuracy_history,
                        'exposure_training_loss_history': exposure_training_loss_history,
                        'exposure_training_accuracy_history': exposure_training_accuracy_history,
                        'exposure_validation_loss_history': exposure_validation_loss_history,
                        'exposure_validation_accuracy_history': exposure_validation_accuracy_history}
    
    '''
    
    with open(history_dict_pkl_path, 'rb') as f:
        history_dict = pickle.load(f)
    

                            
                            
    fig = plt.figure(figsize=(20,20))

    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Training/Validation Loss across Epochs (with EXP)")
    plt.scatter([-0.1], history_dict['exposure_training_loss_history'], c='darkblue', marker="P", label='EXP-train-loss')
    plt.scatter([-0.1], history_dict['exposure_validation_loss_history'], c='orangered', marker="X", label= 'EXP-val-loss')
    plt.plot(range(history_dict['epochs']), history_dict['training_loss_history'], label='train-loss')
    plt.plot(range(history_dict['epochs']), history_dict['validation_loss_history'], label= 'validation-loss')
    plt.ylabel('Training/Validation Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best')

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("Training/Validation Accuracy across Epochs (with EXP)")
    plt.scatter([-0.1], history_dict['exposure_training_accuracy_history'],c='darkblue', marker="P", label='EXP-train-accuracy')
    plt.scatter([-0.1], history_dict['exposure_validation_accuracy_history'], c='orangered', marker="X", label= 'EXP-val-accuracy')
    plt.plot(range(history_dict['epochs']), history_dict['training_accuracy_history'], label='train-accuracy')
    plt.plot(range(history_dict['epochs']), history_dict['validation_accuracy_history'], label= 'validation-accuracy')
    plt.ylabel('Training/Validation accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    
    
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Training/Validation Loss across Epochs")
    plt.plot(range(history_dict['epochs']), history_dict['training_loss_history'], label='train-loss')
    plt.plot(range(history_dict['epochs']), history_dict['validation_loss_history'], label= 'validation-loss')
    plt.ylabel('Training/Validation Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best')

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("Training/Validation Accuracy across Epochs")
    plt.plot(range(history_dict['epochs']), history_dict['training_accuracy_history'], label='train-accuracy')
    plt.plot(range(history_dict['epochs']), history_dict['validation_accuracy_history'], label= 'validation-accuracy')
    plt.ylabel('Training/Validation accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    
    fig.suptitle(f"Training Run Loss/Accuracy History {history_dict_pkl_path.split('/')[-1].split('training')[0]}")
    
    plt.show()



def plot_saved_montage(image_path):
    '''
    helper function to take a montage saved to disk and plot it
    since every other way apparently doesn't work
    '''
    im = PIL.Image.open(image_path, 'r')
    
    #hard coded crop :(
    left = 1980
    top = 200
    right = 3120
    bottom = 2600
    
    im = im.crop((left, top, right, bottom))
    
    fig = plt.figure(figsize=(70., 45.))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(im)
    plt.show()



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


def plot_incorrects_grid(imgs, predicted_labels, groundtruth_labels, plot_id_GT, fig_title, model_pth_name, plot_id_dict, pred_confs):
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
     

    #output
    fig.text(x=0.505, y=0.915, s=fig_title, fontsize=28, ha="center", transform=fig.transFigure)
    fig.text(x=0.5, y=0.90, s=f"model: {model_pth_name}", fontsize=18, ha="center", transform=fig.transFigure)

    plt.show()


def collect_fifty_random_preds(data_csv, saved_model, find_incorrects=True, collect_class=None):

    #collect test dataset and create loader iterable-object
    dataset_path = '/pless_nfs/home/matthewrberning/multi-year-cult-class/data/preprocessed/'

    transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])

    data = WheatAwnDataset(csv_filepath=data_csv, dataset_dir=dataset_path, transform=transform)
    dataloader = DataLoader(data, batch_size=10, shuffle=True)



    #set model mode
    saved_model.eval()

    imgs = []
    predicted_labels = []
    groundtruth_labels = []
    plot_id_GT = []
    pred_confs = []

    b = 0 #breaker

    #make sure to not accumulate gradients
    with torch.no_grad():
        
        progress_bar = tqdm(dataloader, total=int(len(dataloader)), desc=f"Progress: ")

        for step, data in enumerate(progress_bar):

            images, labels, plot_ids = data[0], data[1], data[2]

#         for images, labels, plot_ids in dataloader:
            
            if b==50:
                break

            #send the tensors to the device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            #images = images.float() #uncomment if using read_image() from torch
            outputs = saved_model(images)

            #find the predicted classes indicies
            _, preds = torch.max(outputs, 1)
            
            #collect the 'confidence' using softmax
            soft_preds = torch.softmax(outputs, 1)



            for index, prediction in enumerate(preds):
                #are we looking for wrong predictions?
                if find_incorrects:
                    if prediction == labels[index]:
                        #print("correct!")
                        continue

                    else:
                        #when looking for mistakes in a specific class
                        if collect_class:
                            if prediction == int(collect_class):
                                #print("found missed prediction class: ", prediction)
                                imgs.append(tensor_to_image(images[index]))
                                predicted_labels.append(prediction.detach().cpu().numpy())
                                groundtruth_labels.append(labels[index].detach().cpu().numpy())
                                plot_id_GT.append(plot_ids[index].detach().cpu().numpy())
                                pred_confs.append(soft_preds[index].cpu().numpy()[prediction])
                                b+=1 #add to breaker when we've accumulated another false preditcion

                                if b==50:
                                    #we've collected 50 (b == 50) to plot
                                    break
                                    
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
                                break

                else:
                    if prediction == labels[index]:
                        #when looking for mistakes in a specific class
                        if collect_class:
                            if prediction == int(collect_class):
                                #print("correct!")
                                imgs.append(tensor_to_image(images[index]))
                                predicted_labels.append(prediction.detach().cpu().numpy())
                                groundtruth_labels.append(labels[index].detach().cpu().numpy())
                                plot_id_GT.append(plot_ids[index].detach().cpu().numpy())
                                pred_confs.append(soft_preds[index].cpu().numpy()[prediction])
                                b+=1 #add to breaker when we've accumulated another CORRECT preditction

                                if b==50:
                                    #we've collected 50 (b == 50) to plot
                                    break
                        else:
                            #print("correct!")
                            imgs.append(tensor_to_image(images[index]))
                            predicted_labels.append(prediction.detach().cpu().numpy())
                            groundtruth_labels.append(labels[index].detach().cpu().numpy())
                            plot_id_GT.append(plot_ids[index].detach().cpu().numpy())
                            pred_confs.append(soft_preds[index].cpu().numpy()[prediction])
                            b+=1 #add to breaker when we've accumulated another CORRECT preditction

                            if b==50:
                                #we've collected 50 (b == 50) to plot
                                break
                            

                    else:
                        #print("false!")
                        continue
                        
                    

    print("collected: ", len(imgs), len(predicted_labels), len(groundtruth_labels), len(plot_id_GT), len(pred_confs))
    
    return imgs, predicted_labels, groundtruth_labels, plot_id_GT, pred_confs





def build_conditional_montages(data_csv,
                              batch_size, 
                              fig_title, 
                              model_pth,
                              pkl_file_path,
                              model_name, 
                              end_after=None, 
                              find_incorrects=True, 
                              save_dir='./data/montages',
                              verbose=False,
                              collect_class=None):
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

        model_name : string, required
            the model (currently 'vgg16' or 'resnet50') architecture to use when loading

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

        save_dir : string, optional (default is './data/montages')
            location to save the finished montages

        verbose : bool, optional (default is False)
            control amount of output regarding saving, etc.

        collect_class : string, optional (default is None)
            option to control the collection of montages of a specific class only (i.e. '1' or '0')
            
    """
    print("\n\nbuilding montages...")

    #load up the model
    print(f"\nloading model from: {model_pth}")
    saved_model = Model(model_name).construct_model(verbose=False)
    saved_model.load_state_dict(torch.load(model_pth))
    
    #send the model to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nsending model to device: ", device)
    saved_model = saved_model.to(device)

    #collect test dataset and create loader iterable-object
    transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])
    data = WheatAwnDataset(csv_filepath=data_csv, transform=transform)
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
    print("collect_class: ", collect_class)
    print("verbosity: ", verbose)

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
                    #are we looking for wrong predictions?
                    if find_incorrects:
                        if prediction == labels[index]:
                            #print("correct!")
                            continue

                        else:
                            #when looking for mistakes in a specific class
                            if collect_class:
                                if prediction == int(collect_class):
                                    #print("found missed prediction class: ", prediction)
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
                            #when looking for mistakes in a specific class
                            if collect_class:
                                if prediction == int(collect_class):
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