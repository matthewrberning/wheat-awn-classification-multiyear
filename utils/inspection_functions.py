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
from input_validation import yesno, open_dict_from_pkl
from tensor_operations import tensor_to_image

import PIL
import GPUtil
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import confusion_matrix
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def poll_plots(data_csv, 
               plot_id_dict, 
               date_dict, 
               saved_model, 
               device, 
               confusion_matrix_title, 
               voting_method,
               batch_size=64, 
               verbose=False):
    """
    function to determine a trained model's prediction through majority-rule voting,
    either across all examples for a specific plot_id or using only those examples
    that were collected during a single data-collection episode (a single flight)

    Keyword Arguments: 
        data_csv : string, required 
            the .CSV file containing the dataset description (labels, images, plot_ids, date)
            that can be interpreted by the dataloader/model

        plot_id_dict : string, required 
            the pickel file that determines the relationship between plot_id 
            and numeric value that can be understood by the torch dataloader

        date_dict : string, required 
            the pickel file that determines the relationship between the date 
            of capture and the numeric value that can be understood by the torch dataloader

        saved_model : string, required 
            the path to the model pth file to be used to make the predictions

        device : string, required 
            the device to use ('cuda'/'cuda:0' or 'cpu')


        confusion_matrix_title : string, required 
            the title of the confusion matrix to be output

        voting_method : string, required 
            the method of counting the votes, either 'plot' or 'date'

        verbose : bool, optional (default is False)
            control amount of output regarding saving, etc.
            
    """

    #create validation transforms
    transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])

    #build torch dataset/dataloader
    data = WheatAwnDataset(csv_filepath=data_csv, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    #>>>>COLLECT THE VOTES

    #create dictionaries to hold all predictions for a plot
    vote_dict = {}
    #and all ground truth lables
    GT_dict = {}

    progress_bar = tqdm(dataloader, total=int(len(dataloader)), desc=f"Progress: ")

    for step, data in enumerate(progress_bar):

        images, labels, plot_ids, dates = data[0], data[1], data[2], data[3]

        #send the tensors to the device (GPU)
        images = images.to(device)
        labels = labels.to(device)

        #images = images.float() #uncomment if using read_image() from torch
        outputs = saved_model(images)

        #find the predicted classes indicies
        _, preds = torch.max(outputs, 1)

        for index, plot_id in enumerate(plot_ids):

            #make the numeric plot_id a string 
            #so it can be a key in a dictionary
            plot_id = str(plot_id.item())

            #make the date of the photo's collection a string too
            date = str(dates[index].item())



            #find all the current keys at each step
            current_keys = list(vote_dict.keys())


            if voting_method == 'date':

                plot_id_date_string = plot_id + "_" + date


                #if the plot and date combination-string for the image is already in 
                #the dictionary then just add the prediction
                if plot_id_date_string in current_keys:
                    vote_dict[plot_id_date_string].append(preds[index].item())
                    GT_dict[plot_id_date_string].append(labels[index].item())

                #otherwise make a new key for the combo in the dictionary
                #and add a list as the value, with the elem being the prediction
                else:
                    vote_dict[plot_id_date_string]=[preds[index].item()]
                    GT_dict[plot_id_date_string]=[labels[index].item()]  

            elif voting_method == 'plot':

                #if the plot for the image is already in the dictionary
                #then just add the prediction
                if plot_id in current_keys:
                    vote_dict[plot_id].append(preds[index].item())
                    GT_dict[plot_id].append(labels[index].item())

                #otherwise make a new key in the dictionary for the plot
                #and add a list as the value, with the elem being the prediction
                else:
                    vote_dict[plot_id]=[preds[index].item()]
                    GT_dict[plot_id]=[labels[index].item()]

    #>>>>>COUNT THE VOTES  

    #collect a list of the numeric representations of the plot_id's            
    # numeric_plot_ids_list = list(plot_votes.keys())
    #collect a list of all of the keys for all of the votes in the dictionary
    #these can be "plot_id" + "_" + "date" or just "plot_id" depending on the 
    #vote type
    voting_keys = list(vote_dict.keys())

    #count the votes/predictions and compare
    #with the ground truth for conf. matrix
    awns = 0
    awnless = 0
    awn_corrects = 0
    awnless_corrects = 0
    total_corrects = 0

    #make lists of gt's and preds as well for the confoosion matrix
    predictions = []
    ground_truths = []

    #collect mistakes 
    mistakes_dict = {}
    #{'plot_id':{'gt':0, 'pred':1, 'vote_pct':0.734}}
    #{'plot_id':{'date':20,gt':0, 'pred':1, 'vote_pct':0.734}}  

    for key in voting_keys:

        #find the class from the first label in the ground 
        #truth dict for the vote-block in question
        gt_class = GT_dict[key][0]

        #add the GT to the confusion matrix list
        ground_truths.append(gt_class)

        if gt_class == 0:
            awns+=1
        else:
            awnless+=1

        if voting_method == 'plot':
            if verbose: print("plot_id number: ", key)
            if verbose: print("plot Ground Truth class: ", gt_class)

        if voting_method == 'date':
            if verbose: print("plot_id number, date number: ", key.split("_"))
            if verbose: print("Ground Truth class: ", gt_class)

        #calculate the model's vote for the plot by summing the lables
        #the awnless label is 1 so if more than 50% of the plot is 1's then
        #the model voted for it to be awnless, else it's awned - 0's
        #(but there's also the posibility of a tie)
        vote = sum(vote_dict[key])/len(vote_dict[key])

        if vote  >= 0.5:

            #add to confusion matrix
            predictions.append(1)

            if verbose: print("model predicted (voted for) class: 1")

            if gt_class == 1:
                #correct! 
                awnless_corrects+=1

            else:
                #it's a mistake! 
                mistakes_dict[key] ={'gt':gt_class, 'pred':1, 'vote':vote}


        elif vote < 0.5:

            #add to confusion matrix
            predictions.append(0)

            if verbose: print("model predicted (voted for) class: 0")

            if gt_class == 0:
                #correct!
                awn_corrects+=1

            else:
                #it's a mistake! 
                mistakes_dict[key] ={'gt':gt_class, 'pred':0, 'vote':vote}

        else:
            print("TIE!???!!")
            if voting_method == 'plot':
                print("--> plot_id number: ", key)
                print("--> plot Ground Truth class: ", gt_class)
            if voting_method == 'date':
                print("--> plot_id + _ + date numbers: ", key)
                print("--> Ground Truth class: ", gt_class)

        if voting_method == 'plot':
            #collect the string representation of the plot id from the loaded json
            plot_id_string = list(plot_id_dict.keys())[list(plot_id_dict.values()).index(int(key))]

            if verbose: print("plot_id string: ", plot_id_string, "\n")

        if voting_method == 'date':
            #collect the string representation of the plot_id and the date from the loaded json
            plot_id_string = list(plot_id_dict.keys())[list(plot_id_dict.values()).index(int(key.split("_")[0]))]
            date_string = list(date_dict.keys())[list(date_dict.values()).index(int(key.split("_")[1]))]

            if verbose: print("plot_id string: ", plot_id_string, "\n")
            if verbose: print("plot_id string: ", date_string, "\n")

    #report
    print("total awned plots: ",awns,"\ntotal awnless plots: ", awnless)
    print("awned pct. correct: ", round((awn_corrects/awns), 2)*100, "%")
    print("awnless pct. correct: ", round((awnless_corrects/awnless), 2)*100, "%")
    print("total pct. correct: ", round(((awnless_corrects+awn_corrects)/(awnless+awns)), 2)*100, "%")

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
    plt.title(confusion_matrix_title, fontsize=18)
    plt.show()


    return vote_dict, GT_dict, mistakes_dict
    


def collect_poll_mistakes(mistakes_dict,
                          voting_method,
                          data_csv,
                          plot_id_dict,
                          date_dict,
                          saved_model,
                          device,
                          batch_size=128,
                          collect_fifty=False):

    """
    this function takes in the mistakes dictionary (mistakes_dict) that is output
    by the poll_plots() function. It then reconstructs a dataloader and iterates
    through it to find the images that contributed to the plot being voted into
    the wrong class, also collected is the ground truth label, the plot id and
    the confidence in the prediction, then everything wrong is packaged up into
    another dictionary that may be fed into the montage plotting function, there
    is also a flag to collect only fifty misses (just enough for the plotting fn)

    Keyword Argumens: 
        mistakes_dict : dictionary object, required 
            the mistakes_dict object returned from the poll_plots() fn with the plot_id
            in numeric representation as the primary set of keys, the dict has the following 
            format: {'plot_id':{'gt': 1, 'pred': 0, 'vote': 0.38402457757296465}}

        voting_method : string, required
            the voting method used when constructing the mistakes_dict (the one used
            with the corresponding run of the poll_plots() fn)

        plot_id_dict : string, required 
            the pickel file that determines the relationship between plot_id 
            and numeric value that can be understood by the torch dataloader

        date_dict : string, required 
            pickel file that determines the relationship between the date 
            of capture and the numeric value that can be understood by the torch dataloader

        saved_model : torch.model(), required 
            the loaded/instantiated model to be used        

        device : torch.device(), required 
            the device to use ('cuda'/'cuda:0' or 'cpu')

        batch_size : int, optional (default is 128) 
            the number of images to be returned each iteration of the dataloader

        collect_fifty : bool optional (default is False) 
            a flag to determine if the function will only collect 50 examples for
            each incorretly voted plot or plot-date combo

    """
    
    #collect all of the plot_id's (keys) of the mistakes dict
    mistakes_keys = list(mistakes_dict.keys())
    
    #set model mode
    saved_model.eval()
        
    collected_mistakes_dict = {}
    #{'plot_id':{'imgs':[],'predicted_labels':[], 'groundtruth_labels':[], 'plot_id_GT':[], 'pred_confs':[]}}
    #{'plot_id_date':{'imgs':[],'predicted_labels':[], 'groundtruth_labels':[], 'plot_id_GT':[], 'pred_confs':[]}}
        
    #create validation transforms
    transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])
    
    #build torch dataset/dataloader
    data = WheatAwnDataset(csv_filepath=data_csv, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    #make sure to not accumulate gradients
    with torch.no_grad():

        for key in mistakes_keys:
            
            print("looking for: ", key, "mistakes_dict: ",mistakes_dict[key])
            
            if collect_fifty:
                #if we're only finding 50 then break after finding 50 examples
                b=0
            
            #collect the relevant data for each missed prediction in lists
            imgs = []
            predicted_labels = []
            groundtruth_labels = []
            plot_id_GT = []
            pred_confs = []
            
            #track correct and incorrect predictions for ratio
            corrects = 0
            incorrects = 0

            progress_bar = tqdm(dataloader, total=int(len(dataloader)), desc=f"Progress: ")

            for step, data in enumerate(progress_bar):
                
                if collect_fifty:
                    if b==50:
                        print("-50 found for key: ", key)
                        break

                images, labels, plot_ids, dates = data[0], data[1], data[2], data[3]

                #send the tensors to the device (GPU)
                images = images.to(device)
                labels = labels.to(device)

                #images = images.float() #uncomment if using read_image() from torch
                outputs = saved_model(images)

                #find the predicted classes indicies
                _, preds = torch.max(outputs, 1)

                #collect the 'confidence' using softmax
                soft_preds = torch.softmax(outputs, 1)

                for index, plot_id in enumerate(plot_ids):
                    if voting_method == 'plot':
                        #make the numeric plot_id a string 
                        #so it can be a key in a dictionary
                        plot_id_str = str(plot_id.item())

                        #if we've found the right plot and it's a bad prediction, add it
                        if plot_id_str == key and preds[index] != labels[index]:
                            incorrects+=1

                            imgs.append(tensor_operations.tensor_to_image(images[index]))
                            predicted_labels.append(preds[index].detach().cpu().numpy())
                            groundtruth_labels.append(labels[index].detach().cpu().numpy())
                            plot_id_GT.append(plot_id)
                            pred_confs.append(soft_preds[index].cpu().numpy()[preds[index].detach().cpu().item()])
                            
                            if collect_fifty:
                                b+=1 #add to breaker when we've accumulated another false preditcion

                                if b==50:
                                    #we've collected 50 (b == 50)
                                    print("50 found for key: ", key)
                                    break
                                    
                        elif plot_id_str == key and preds[index] == labels[index]:
                            corrects+=1
                                    
                    if voting_method == 'date':
                        #make the numeric plot_id a string 
                        #so it can be a key in a dictionary
                        plot_id_str = str(plot_id.item())

                        #make the date of the photo's collection a string too
                        date_str = str(dates[index].item())
                        
                        #remeber the key is the plot_id and the date with a '_' between them
                        key_str = plot_id_str + "_" + date_str
                        
                        #check for the key and a miss
                        if key_str == key and preds[index] != labels[index]:
                            incorrects+=1
                            
                            #if we've found the correct key then get the right datums for the key
                            imgs.append(tensor_operations.tensor_to_image(images[index]))
                            predicted_labels.append(preds[index].detach().cpu().numpy())
                            groundtruth_labels.append(labels[index].detach().cpu().numpy())
                            plot_id_GT.append(plot_id)
                            pred_confs.append(soft_preds[index].cpu().numpy()[preds[index].detach().cpu().item()])
                            
                            if collect_fifty:
                                b+=1 #add to breaker when we've accumulated another false preditcion

                                if b==50:
                                    #we've collected 50 (b == 50)
                                    print("50 found for key: ", key)
                                    break
                            
                        elif key_str == key and preds[index] == labels[index]:
                            corrects+=1
    
            #return the collection for that key
            collected_mistakes_dict[key]={'imgs':imgs,
                                          'predicted_labels':predicted_labels, 
                                          'groundtruth_labels':groundtruth_labels, 
                                          'plot_id_GT':plot_id_GT, 
                                          'pred_confs':pred_confs,
                                          'corrects':corrects,
                                          'incorrects':incorrects}
            
            #report
            total_collected = len(collected_mistakes_dict[key]['predicted_labels'])
            print(f"finished collection for key {key}, total for key: {total_collected}")
                        
    return collected_mistakes_dict


def get_montages(model_name):
    '''
    helper function to collect a list of montages made using a specific 
    model_name in the montage dir (hard coded as "../data/montages")

    Keyword Arguments: 
        model_name: string, required 
            the unique model name/timestamp to find in the collection of
            montage files

    '''

    search_dir = "../data/montages"
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
    
    transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])

    data = WheatAwnDataset(csv_filepath=data_csv, transform=transform)
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
    '''
    helper function to take a history dictionary pickel file and plot it
    
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


def collect_fifty_random_preds(data_csv, saved_model, device, find_incorrects=True, collect_class=None):


    transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])

    data = WheatAwnDataset(csv_filepath=data_csv, transform=transform)
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