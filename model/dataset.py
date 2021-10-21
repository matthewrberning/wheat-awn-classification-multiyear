import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image

class WheatAwnDataset(Dataset):
	""" dataset class for awn/awnless wheat data """

	def __init__(self, csv_filepath, dataset_dir, transform=None, verbose=True):
		"""	
		Keyword Argumens: 
			csv_filepath : str
				the path to the .csv dataset definition (key_file or similar)
			dataset_dir : str
				the path to the directory where the images are located
			transform : callable, optional (default is None)
				transform.compose() object with normalizations/data augmentation
			verbose : bool, optional (default is True)
				set the verbosity of the dataset construction
		"""

		#make a dataframe from the csv file
		self.dataset_df = pd.read_csv(csv_filepath)

		if verbose: print(f"data file {csv_filepath.split('/')[0]} contains {len(self.dataset_df)} rows")

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
		#note: class label (awned: 0, awnless: 1)
		image_path = self.dataset_df.iloc[idx, 0]
		label = self.dataset_df.iloc[idx, 1]

		#read the image with torch (returns a uint8 tensor)
		#no transforms.ToTensor() needed
		#https://github.com/pytorch/vision/issues/2788
		# image = read_image(image_path)

		#open image with PIL
		image = Image.open(image_path)

		#transform (data augmentation)
		if self.transform:
			image = self.transform(image)

		return image, label



