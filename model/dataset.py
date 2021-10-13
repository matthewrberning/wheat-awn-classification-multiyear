import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class WheatAwnDataset(Dataset):
	""" dataset class for awn/awnless wheat data """

	def __init__(self, csv_filepath, dataset_dir, transform=None):
		"""	
		Keyword Argumens: 
			csv_filepath : str
				the path to the .csv dataset definition (key_file or similar)
			dataset_dir : str
				the path to the directory where the images are located
			transform : callable, optional (default is None)
				transform.compose() object with normalizations/data augmentation
		"""

		#make a dataframe from the csv file
		self.dataset_df = pd.read(csv_filepath)

		#set the path for the root folder to the images of the dataset
		self.dataset_dir = dataset_dir

		#transforms object
		self.transform = transform

	def __len__(self):
		""" get the size of the dataframe """

		return len(self.dataset_df)

	def __getitem__(self, idx):
		""" collect images from the dataset """

		#find the image path and label in the df
		image_path = self.dataset_df.iloc[idx, 0]
		label = self.dataset_df.iloc[idx, 1]

		#read the image with torch
		image = read_image(image_path)

		#transform (data augmentation)
		if self.transform:
			image = self.transform(image)

		return image, label



