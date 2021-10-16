import torch
import torch.nn as nn
from torchvision import models

class Model():
	"""class to hold model objects"""
	def __init__(self, model_name='vgg16'):
		"""
		Keyword Argumens: 
			model_name : str, optional (default is 'vgg16')
				placeholder for future testing of multiple architectures
				for fine-tuining 
		"""

		self.model_name = model_name

		if model_name == 'vgg16':

			#collect the VGG16 model architecture and weights (trained on
			#the 14million-strong ImageNet dataset) from Pytorch's Model Zoo
			#source: https://pytorch.org/vision/stable/models.html
			#paper: https://arxiv.org/abs/1409.1556
			self.model = models.vgg16(pretrained=True)
		
	def construct_model(self, verbose=False):
		"""
		function to replace the last fully connected
		layer of the model (1000 nodes for the
		1000 classes of ImageNet) with the custom layer
		we need to solve our problem

		Note: no activation function on the final linear
		layer since we will be using nn.CrossEntropyLoss() 
		as our loss function an that depends upon the raw
		score of the nodes being output at the end of the 
		forward pass

		Keyword Argumens: 
			verbose : bool, optional (default is False)
				control the amount of output when building
				i.e. weather to print the model
		"""

		#go through the feature extraction portion of the
		#model and ensure that they will not experience weight
		#updates during training
		for param in self.model.features.parameters():
			param.requires_grad = False

		#collect the number of nodes to connect to in the
		#second to last linear layer (vgg16 it's layer 6)
		input_features = self.model.classifier[-1].in_features

		#construct our custom classification layer
		terminal_layer = nn.Linear(input_features, 2)

		#replace the original final linear layer with ours
		self.model.classifier[-1] = terminal_layer

		#check for verbosity
		if verbose: print(self.model)

		return self.model

# print("running.")

# m = Model().construct_model(verbose=True)

# print('done.')
