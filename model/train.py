import os 
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #set the GPU to use during training

import sys
import time


from model import Model
from dataset import WheatAwnDataset


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

def expose(model, epoch, dataloader, device, criterion, optimizer):
    print(f"EXPOSE      epoch: {epoch}")
    
    #track the loss across the epoch
    epoch_loss = 0.0

    #track the correct predictions
    corrects = 0.0

    #set model mode
    model.eval()

    #make sure to not accumulate gradients
    with torch.no_grad():

        #create progress bar with tqdm
        progress_bar = tqdm(dataloader, total=int(len(dataloader)), desc='EXPOSURE Progress: ')

        for step, data in enumerate(progress_bar):

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


    #calculate the total loss across the iterations of the loader
    epoch_loss = epoch_loss/len(dataloader)

    #calculate the accuracy 
    accuracy = (corrects/len(dataloader.dataset))*100

    #format the epoch loss/accuracy to look nice
    epoch_loss = "{:7.5f}".format(epoch_loss)
    accuracy = "{:5.2f}".format(accuracy)

    print(f"      ---> loss: {epoch_loss} accuracy: {accuracy}")

    return epoch_loss, accuracy





def train(model, epoch, dataloader, device, criterion, optimizer):
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


    #calculate the total loss across the iterations of the loader
    epoch_loss = epoch_loss/len(dataloader)

    #calculate the accuracy 
    accuracy = (corrects/len(dataloader.dataset))*100

    #format the epoch loss/accuracy to look nice
    epoch_loss = "{:7.5f}".format(epoch_loss)
    accuracy = "{:5.2f}".format(accuracy)

    print(f"      ---> loss: {epoch_loss} accuracy: {accuracy}")

    return epoch_loss, accuracy

def validate(model, epoch, dataloader, device, criterion, optimizer):
    print(f"[validate]  epoch: {epoch}")

    #track the loss across the epoch
    epoch_loss = 0.0

    #track the correct predictions
    corrects = 0.0

    #set model mode
    model.eval()

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
    epoch_loss = "{:7.5f}".format(epoch_loss)
    accuracy = "{:5.2f}".format(accuracy)

    print(f"      ---> loss: {epoch_loss} accuracy: {accuracy}")

    return epoch_loss, accuracy

def main():

    print('running...\n\n')
    current_time = time.strftime("%Y-%m-%d-%H_%M_%S")

    #set the 'device' customary var for the GPU (or CPU if not available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("(using device: ", device,")\n\n")


    #construct the datasets
    dataset_path = '/pless_nfs/home/matthewrberning/multi-year-cult-class/data/preprocessed/'

    #training data
    print("building training set...")
    train_data_csv = '/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/data/2019_train_awns_oversampled.csv'
    train_transform = transforms.Compose([transforms.RandomCrop((224,224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor()])
    
    training_data = WheatAwnDataset(csv_filepath=train_data_csv, dataset_dir=dataset_path, transform=train_transform)
    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

    #validation data
    print("building validation set...")
    validation_data_csv = '/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/data/2019_val_awns_oversampled.csv'
    validation_transform = transforms.Compose([transforms.ToTensor()])
    
    validation_data = WheatAwnDataset(csv_filepath=validation_data_csv, dataset_dir=dataset_path, transform=validation_transform)
    validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=True)

    #build the model
    model = Model().construct_model(verbose=False)

    model = model.to(device)

    #loss function
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

    #set up lists to track training progress
    training_loss_history, training_accuracy_history = [], []
    validation_loss_history, validation_accuracy_history = [], []

    #train the model across epochs
    epochs = 10

    print(f"\n\ntraining across {epochs} epochs\n\n")

    for epoch in range(epochs):

        #make sure to expose the model first, with just the raw initilizations
        if epoch == 0:
            loss, accuracy = expose(model, epoch-1, train_dataloader, device, criterion, optimizer)
            training_loss_history.append(loss)
            training_accuracy_history.append(accuracy)

            loss, accuracy = expose(model, epoch-1, validation_dataloader, device, criterion, optimizer)
            validation_loss_history.append(loss)
            validation_accuracy_history.append(accuracy)

        #train
        loss, accuracy = train(model, epoch, train_dataloader, device, criterion, optimizer)
        training_loss_history.append(loss)
        training_accuracy_history.append(accuracy)

        #validate
        val_loss, val_accuracy = validate(model, epoch, validation_dataloader, device, criterion, optimizer)
        validation_loss_history.append(val_loss)
        validation_accuracy_history.append(val_accuracy)
        
        if epoch%1 == 0 and epoch != 0 and epoch != epochs:
            #save the model (parameters only) every 5 epochs
            print(f"\n\nsaving model at epoch {epoch} path:\n     ../runs/{current_time}_model_epoch-{epoch}.pth")
            torch.save(model.state_dict(), f"/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/runs/{current_time}_model_epoch-{epoch}_val-acc-{val_accuracy:.3f}.pth")
    
    #save final model and plot training history
    torch.save(model.state_dict(), f"/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/runs/{current_time}_model_epoch-{epoch}_val-acc-{val_accuracy:.3f}.pth")
    
    print("state_dict saved.. plotting accuracy/loss history")

    fig = plt.figure(figsize=(20,8))

    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Training/Validation Loss across Epochs")
    plt.plot(range(epochs+1), training_loss_history, label='train-loss')
    plt.plot(range(epochs+1), validation_loss_history, label= 'validation-loss')
    
    plt.ylabel('Training/Validation Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best')

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Training/Validation Accuracy across Epochs")
    plt.plot(range(epochs+1), training_accuracy_history, label='train-accuracy')
    plt.plot(range(epochs+1), validation_accuracy_history, label= 'validation-accuracy')
    plt.ylabel('Training/Validation accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    
    fig.suptitle(f"Training Run Loss/Accuracy History {current_time}")
    fig.savefig(f"/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/runs/{current_time}_loss_accuracy-plot.jpg")

    print("\n...terminating")

if __name__ == '__main__':
    main()

