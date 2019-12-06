# Pre-trained resnet50 model for feature extraction

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam
import keras
import cv2
import glob
import os
import numpy as np 

HEIGHT = 300
WIDTH  = 300
IMAGE_PATH = "/home/kishor/GWM/Meta_Cognition_Experiments/1_Clustering/Dataset/image_dataset/"


def build_finetune_model(base_model, dropout, fc_layers, num_classes):

	return 0


def extract_features(path, resnet_model):

	resnet_feature_list = []
	counter = 0

	os.chdir(IMAGE_PATH)
	imagelist = glob.glob("*.jpg")

	for image in imagelist:
		print(image)
		image = cv2.imread(image)
		image = cv2.resize(image, (HEIGHT, WIDTH))
		image = preprocess_input(np.expand_dims(image.copy(), axis = 0))
		resnet_features = resnet_model.predict(image)
		resnet_features_np = np.array(resnet_features)
		resnet_feature_list.append(resnet_features_np)
		counter += 1
		#if (counter == 10):
		#	break
	return np.array(resnet_feature_list)


if __name__ == "__main__":

	print("Lets extract the features")

	curr_dir = os.getcwd()

	base_model = ResNet50(weights = "imagenet",
						  include_top = False,
						  input_shape = (HEIGHT, WIDTH, 3))

	for layer in base_model.layers:
		layer.trainable = False

	feature_vectors = extract_features(IMAGE_PATH, base_model)

	os.chdir(curr_dir)
	np.save("feature_vector", feature_vectors)

	#print(feature_vectors)

	#finetune_model = build_finetune_model(base_model,
	#									  dropout = dropout, 
	#									  fc_layers = FC_LAYERS)