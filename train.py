#matthew berning - GWU, 2021
import os 
import sys
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" #set the GPU to use during training

import time
import argparse

from model.model import Model
from data.dataset import WheatAwnDataset
from utils.input import yesno


import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import GPUtil
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

def expose(model, epoch, dataloader, device, criterion, optimizer, class_counts, mode):
    print(f"EXPOSE      epoch: {epoch}")
    
    #track the loss across the epoch
    epoch_loss = 0.0

    #track the correct predictions
    corrects = 0.0

    #set model mode
    model.eval()

    if mode == 'val':
        majority_count = 0

    #make sure to not accumulate gradients
    with torch.no_grad():

        #create progress bar with tqdm
        progress_bar = tqdm(dataloader, total=int(len(dataloader)), desc='EXPOSURE Progress: ')

        for step, data in enumerate(progress_bar):

            if mode == 'val':
                if majority_count == class_counts[0]:
                    break

            #unpack the data from the progress bar
            images, labels = data[0], data[1]

            #send the tensors to the device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            #images = images.float() #uncomment if using read_image() from torch
            outputs = model(images) #float?

            #calculate the loss
            loss = criterion(outputs, labels)

            #add to the loss accumulated over the epoch
            epoch_loss += loss.item()

            #find the predicted classes indicies
            _, preds = torch.max(outputs, 1)

            #track the correct predictions (.item()to collect just the value)
            corrects += torch.sum(preds == labels.data).item()

            if mode == 'val' and labels[0].item() == 0:
                print("adding to majority count!!!!")
                majority_count += 1


    #calculate the total loss across the iterations of the loader
    epoch_loss = epoch_loss/len(dataloader)

    #calculate the accuracy 
    accuracy = (corrects/len(dataloader.dataset))*100

    #format the epoch loss/accuracy to look nice
    epoch_loss_str = "{:7.5f}".format(epoch_loss)
    accuracy_str = "{:5.2f}".format(accuracy)

    print(f"      ---> loss: {epoch_loss_str} accuracy: {accuracy_str}")

    return epoch_loss, accuracy





def train(model, epoch, dataloader, device, criterion, optimizer, scheduler):
    print(f"[train]     epoch: {epoch}")

    #set model mode
    model.train()

    #track the loss across the epoch
    epoch_loss = 0.0

    #track the correct predictions
    corrects = 0.0

    #create progress bar with tqdm
    progress_bar = tqdm(dataloader, total=int(len(dataloader)), desc='[training] Progress: ')

    for step, data in enumerate(progress_bar):

        #unpack the data from the progress bar
        images, labels = data[0], data[1]

        #send the tensors to the device (GPU)
        images = images.to(device)
        labels = labels.to(device)
        
        #images = images.float() #uncomment if using read_image() from torch
        outputs = model(images)

        #calculate the loss
        loss = criterion(outputs, labels)

        #reset our gradients
        optimizer.zero_grad()

        #propagate backward
        loss.backward()

        #update weights
        optimizer.step()

        #add to the loss accumulated over the epoch
        epoch_loss += loss.item()

        #find the predicted classes indicies
        _, preds = torch.max(outputs, 1)

        #track the correct predictions
        corrects += torch.sum(preds == labels.data).item()

    #step learning rate
    scheduler.step()

    #calculate the total loss across the iterations of the loader
    epoch_loss = epoch_loss/len(dataloader)

    #calculate the accuracy 
    accuracy = (corrects/len(dataloader.dataset))*100

    #format the epoch loss/accuracy to look nice
    epoch_loss_str = "{:7.5f}".format(epoch_loss)
    accuracy_str = "{:5.2f}".format(accuracy)

    print(f"      ---> loss: {epoch_loss_str} accuracy: {accuracy_str}")

    return epoch_loss, accuracy

def validate(model, epoch, dataloader, device, criterion, optimizer, class_counts):
    print(f"[validate]  epoch: {epoch}")

    #track the loss across the epoch
    epoch_loss = 0.0

    #track the correct predictions
    corrects = 0.0

    #set model mode
    model.eval()

    #count up to parity with the minority class
    majority_count = 0

    #make sure to not accumulate gradients
    with torch.no_grad():

        #create progress bar with tqdm
        progress_bar = tqdm(dataloader, total=int(len(dataloader)), desc='[validation] Progress: ')

        for step, data in enumerate(progress_bar):

            #unpack the data from the progress bar
            images, labels = data[0], data[1]

            #send the tensors to the device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            #images = images.float() #uncomment if using read_image() from torch
            outputs = model(images)

            #calculate the loss
            loss = criterion(outputs, labels)

            #add to the loss accumulated over the epoch
            epoch_loss += loss.item()

            #find the predicted classes indicies
            _, preds = torch.max(outputs, 1)

            #track the correct predictions
            corrects += torch.sum(preds == labels.data).item()


    #calculate the total loss across the iterations of the loader
    epoch_loss = epoch_loss/len(dataloader)

    #calculate the accuracy 
    accuracy = (corrects/len(dataloader.dataset))*100
    
    #format the epoch loss/accuracy to look nice
    epoch_loss_str = "{:7.5f}".format(epoch_loss)
    accuracy_str = "{:5.2f}".format(accuracy)

    print(f"      ---> loss: {epoch_loss_str} accuracy: {accuracy_str}")

    return epoch_loss, accuracy

def main(model_name, train_csv_path, val_csv_path, epochs, learning_rate, lr_lambda, batch_size):

    print('\n\nrunning...\n\n')

    #capture the time at the start of the run
    current_time = time.strftime("%Y-%m-%d-%H_%M_%S")

    #make a var for the reoccuring path name
    dir_path = '/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/'

    #set the 'device' customary var for the GPU (or CPU if not available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("(using device: ", device,")\n\n")


    #construct the datasets
    dataset_path = '/pless_nfs/home/matthewrberning/multi-year-cult-class/data/preprocessed/'

    #training data
    print("building training set...")
    train_data_csv = os.path.join(dir_path, train_csv_path)
    print(train_data_csv)
    train_transform = transforms.Compose([transforms.RandomCrop((224,224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor()])
    
    training_data = WheatAwnDataset(csv_filepath=train_data_csv,
                                    dataset_dir=dataset_path,
                                    transform=train_transform)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    #validation data
    print("building validation set...")
    validation_data_csv = os.path.join(dir_path, val_csv_path)
    validation_transform = transforms.Compose([transforms.CenterCrop((224,224)),
                                               transforms.ToTensor()])
    
    validation_data = WheatAwnDataset(csv_filepath=validation_data_csv,
                                      dataset_dir=dataset_path,
                                      transform=validation_transform)

    #find what the number of instances in each class is for the validation set
    #i.e. {0: 1234, 1: 78910}
    class_counts = dict(Counter(validation_data.targets))

    validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=True)

    #build the model
    model = Model(model_name).construct_model(verbose=False)

    model = model.to(device)

    #loss function
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    #set up learning rate scheduler
    lmbda = lambda epoch: lr_lambda ** epoch
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    #set up lists to track training progress
    training_loss_history, training_accuracy_history = [], []
    validation_loss_history, validation_accuracy_history = [], []

    exposure_training_loss_history, exposure_training_accuracy_history = [], []
    exposure_validation_loss_history, exposure_validation_accuracy_history = [], []

    #train the model across epochs
    print(f"\n\ntraining across {epochs} epochs\n\n")

    for epoch in range(epochs):

        #make sure to expose the model first, with just the raw initilizations
        if epoch == 0:
            mode = 'train'
            loss, accuracy = expose(model, epoch-1, train_dataloader, device, criterion, optimizer, class_counts, mode)

            #track the training history
            exposure_training_loss_history.append(loss)
            exposure_training_accuracy_history.append(accuracy)

            mode = 'validate'
            loss, accuracy = expose(model, epoch-1, validation_dataloader, device, criterion, optimizer, class_counts, mode)

            #track the training history
            exposure_validation_loss_history.append(loss)
            exposure_validation_accuracy_history.append(accuracy)

        #train
        loss, accuracy = train(model, epoch, train_dataloader, device, criterion, optimizer, scheduler)

        #track the training history
        training_loss_history.append(loss)
        training_accuracy_history.append(accuracy)

        #validate
        val_loss, val_accuracy = validate(model, epoch, validation_dataloader, device, criterion, optimizer, class_counts)

        #track the training history
        validation_loss_history.append(val_loss)
        validation_accuracy_history.append(val_accuracy)

        #save a dictionary of the current training loss/accuracy history as a pickel
        history_dict = {'date': current_time,
                        'epochs': epoch+1,
                        'training_loss_history':training_loss_history,
                        'training_accuracy_history':training_accuracy_history,
                        'validation_loss_history':validation_loss_history,
                        'validation_accuracy_history':validation_accuracy_history,
                        'exposure_training_loss_history': exposure_training_loss_history,
                        'exposure_training_accuracy_history': exposure_training_accuracy_history,
                        'exposure_validation_loss_history': exposure_validation_loss_history,
                        'exposure_validation_accuracy_history': exposure_validation_accuracy_history}
        
        #overwrite the previous file with the name, also note: using 'wb' so use 'rb' to load
        with open(os.path.join(dir_path, f"runs/{current_time}_training_history.pkl"), 'wb') as f:
            pickle.dump(history_dict, f)
        
        #save the model (parameters only) every so often
        if epoch%1 == 0:
            print(f"\n\nsaving model at epoch {epoch} path:\n     ../runs/{current_time}_model_epoch-{epoch}.pth")
            torch.save(model.state_dict(), os.path.join(dir_path, f"runs/{current_time}_model_epoch-{epoch+1}_val-acc-{val_accuracy:.3f}.pth"))
   
    
    print("\nplotting accuracy/loss history")

    fig = plt.figure(figsize=(20,8))

    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Training/Validation Loss across Epochs")
    plt.scatter([-0.1], exposure_training_loss_history, c='darkblue', marker="P", label='EXP-train-loss')
    plt.scatter([-0.1], exposure_validation_loss_history, c='orangered', marker="X", label= 'EXP-val-loss')
    plt.plot(range(epochs), training_loss_history, label='train-loss')
    plt.plot(range(epochs), validation_loss_history, label= 'validation-loss')
    
    plt.ylabel('Training/Validation Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best')

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Training/Validation Accuracy across Epochs")
    plt.scatter([-0.1], exposure_training_accuracy_history,c='darkblue', marker="P", label='EXP-train-accuracy')
    plt.scatter([-0.1], exposure_validation_accuracy_history, c='orangered', marker="X", label= 'EXP-val-accuracy')
    plt.plot(range(epochs), training_accuracy_history, label='train-accuracy')
    plt.plot(range(epochs), validation_accuracy_history, label= 'validation-accuracy')
    plt.ylabel('Training/Validation accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    
    fig.suptitle(f"Training Run Loss/Accuracy History {current_time}")
    fig.savefig(os.path.join(dir_path, f"runs/{current_time}_loss_accuracy-plot.jpg"))

    print("\n...terminating")


def assign_arguments():
    parser = argparse.ArgumentParser(description="awn/awnless training script using either vgg16 or resnet")
    parser.add_argument('--model_name', type=str, default='vgg16', required=False)
    parser.add_argument('--train_csv_path', type=str, default='data/2019_train_awns_UNDERsampled.csv', required=False)
    parser.add_argument('--val_csv_path', type=str, default='data/2019_val_awns.csv', required=False)
    parser.add_argument('--epochs', type=int, default=10, required=False)
    parser.add_argument('--learning_rate', type=float, default=0.00001, required=False)
    parser.add_argument('--lr_lambda', type=float, default=0.95, required=False)
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    return parser.parse_args()


if __name__ == '__main__':

    #find the correct GPU -and use it!
    deviceIDs = GPUtil.getAvailable(order = 'first', 
                                    limit = 1, 
                                    maxLoad = 0.3, 
                                    maxMemory = 0.3, 
                                    includeNan=False, 
                                    excludeID=[], 
                                    excludeUUID=[])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])
    print("\n\nGPU Chosen: ", str(deviceIDs[0]))

    args = assign_arguments()

    print("\n\n-arguments supplied/defaulted-\n")
    print("    --model_name: ", args.model_name)
    print("    --train_csv_path: ", args.train_csv_path)
    print("    --val_csv_path: ", args.val_csv_path)
    print("    --epochs: ", args.epochs)
    print("    --learning_rate: ", args.learning_rate)
    print("    --lr_lambda", args.lr_lambda)
    print("    --batch_size", args.batch_size,"\n\n")


    if yesno("are the training conditions above correct?"):
        main(model_name=args.model_name, 
             train_csv_path=args.train_csv_path, 
             val_csv_path=args.val_csv_path, 
             epochs=args.epochs,
             learning_rate=args.learning_rate, 
             lr_lambda=args.lr_lambda, batch_size=args.batch_size)

    else:
        sys.exit("\n\n\n...\n\n\n")

