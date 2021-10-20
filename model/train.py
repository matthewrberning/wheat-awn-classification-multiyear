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

        for images, labels in dataloader:

            #send the tensors to the device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            images = images.float()
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

    print(f"      ---> loss: {str(epoch_loss).rjust(7)} accuracy: {accuracy}")

    return epoch_loss, accuracy





def train(model, epoch, dataloader, device, criterion, optimizer):
    print(f"[train]     epoch: {epoch}")

    #set model mode
    model.train()

    #track the loss across the epoch
    epoch_loss = 0.0

    #track the correct predictions
    corrects = 0.0

    for images, labels in dataloader:

        images = images.float()

        #send the tensors to the device (GPU)
        images = images.to(device)
        labels = labels.to(device)

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

    print(f"      ---> loss: {str(epoch_loss).rjust(7)} accuracy: {accuracy}")

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

        for images, labels in dataloader:

            images = images.float()

            #send the tensors to the device (GPU)
            images = images.to(device)
            labels = labels.to(device)

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

    print(f"      ---> loss: {str(epoch_loss).rjust(7)} accuracy: {accuracy}")

    return epoch_loss, accuracy

def main():

	print('training running')

	#set the 'device' customary var for the GPU (or CPU if not available)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	print("using device: ", device)


	#construct the datasets
	dataset_path = '/pless_nfs/home/matthewrberning/multi-year-cult-class/data/preprocessed/'

	#training data
	train_data_csv = '/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/data/2019_train_awns_oversampled.csv'
	train_transform = transforms.Compose([transforms.RandomRotation(0.2),
                                      transforms.RandomCrop((224,224)), 
                                      transforms.RandomHorizontalFlip(), 
                                      transforms.RandomVerticalFlip()])
	training_data = WheatAwnDataset(csv_filepath=train_data_csv, dataset_dir=dataset_path, transform=train_transform)
	train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)

	#tvalidation data
	validation_data_csv = '/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/data/2019_test_awns.csv'
	validation_transform = None
	validation_data = WheatAwnDataset(csv_filepath=validation_data_csv, dataset_dir=dataset_path, transform=validation_transform)
	validation_dataloader = DataLoader(validation_data, batch_size=8, shuffle=True)

	#build the model
	model = Model().construct_model(verbose=True)

	model = model.to(device)

	#loss function
	criterion = nn.CrossEntropyLoss()

	#optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

	#set up lists to track training progress
	training_loss_history, training_accuracy_history = []
	validation_loss_history, validation_accuracy_history = []

	#train the model across epochs
	epochs = 10

	for epoch in range(epochs):

		#make sure to expose the model first, with just the raw initilizations
		if epoch == 0:
			loss, accuracy = expose(model, epoch, train_dataloader, device, criterion, optimizer)
			train_loss_history.append(loss)
			train_accuracy_history.append(accuracy)

			loss, accuracy = expose(model, epoch, validation_dataloader, device, criterion, optimizer)
			validation_loss_history.append(loss)
			validation_accuracy_history.append(accuracy)

		#train
		loss, accuracy = train(model, epoch+1 if epoch == 0 else epoch, train_dataloader, device, criterion, optimizer)
		train_loss_history.append(loss)
		train_accuracy_history.append(accuracy)

		#validate
		loss, accuracy = validate(model, epoch+1 if epoch == 0 else epoch, validation_dataloader, device, criterion, optimizer)
		validation_loss_history.append(loss)
		validation_accuracy_history.append(accuracy)

	#plot results and save
	# plt.plot(range(epochs), train_loss_history, label='train-loss')
	# plt.plot(range(epochs), validation_loss_history, label= 'validation-loss')
	# plt.ylabel('Training/Validation Loss')
	# plt.xlabel('Epochs')
	# plt.legend(loc='best')
	# plt.savefig("/pless_nfs/home/matthewrberning/wheat-awn-classification-multiyear/runs/")


if __name__ == '__main__':
	main()

