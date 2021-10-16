import os 
import sys
import time

from model import Model
from dataset import WheatAwnDataset

from torch.utils.data import Dataloader
from torchvision import transforms

def expose():
	print(f"EXPOSE  epoch: {epoch}")

def train():
	print(f"[train] epoch: {epoch}")

def test(model, epoch):
	print(f"[test]  epoch: {epoch}")

def main():

	print('running')


	#construct the datasets
	dataset_path = '////'

	#training data
	train_data_csv = '////'
	train_transform = transforms.Compose([/////])
	training_data = WheatAwnDataset(csv_filepath=train_data_csv, dataset_dir=dataset_path, transform=train_transform)
	train_dataloader = Dataloader(training_data, batch_size=64, shuffle=True)

	#testing data
	test_data_csv = '////'
	test_transform = transforms.Compose([///])
	testing_data = WheatAwnDataset(csv_filepath=test_data_csv, dataset_dir=dataset_path, transform=test_transform)
	test_dataloader = Dataloader(testing_data, batch_size=64, shuffle=True)

	#build the model
	model = Model().construct_model(verbose=True)

	for epoch in range(epochs):

		#make sure to expose the model first, with just the raw initilizations
		if epoch == 0:
			expose(model, epoch)

		train(model, epoch+1 if epoch == 0 else epoch)
		test(model, epoch+1 if epoch == 0 else epoch)


if __name__ == '__main__':
	main()

