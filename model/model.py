import torch
import torch.nn as nn
from torchvision import models

print("hello")

class Model():
	def __init__(self):
		#collect the VGG16 model architecture and weights (trained on
		#the 14million-strong ImageNet dataset) from Pytorch's Model Zoo
		#source: https://pytorch.org/vision/stable/models.html
		#paper: https://arxiv.org/abs/1409.1556
		self.model = models.vgg16(pretrained=True)
		
	def construct_model(self, verbose=False):
		"""
		function to replace the last fully connected
		layer of the model (1000 nodes for the
		1000 classes of ImageNet with the 

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
		#second to last linear layer
		input_features = self.model.classifier[6].in_features

		#construct our custom classification layer
		terminal_layer = nn.Linear(input_features, 2)

		#replace the original linear layer with ours
		self.model.classifier[6] = terminal_layer

		print(self.model)

		return self.model

m = Model()

model = m.construct_model()

print("\n\n-params-\n\n",model)