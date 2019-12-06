# Reference : https://medium.com/@franky07724_57962/using-keras-pre-trained-models-for-feature-extraction-in-image-clustering-a142c6cdf5b1
# IMP - https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/


from keras.preprocessing import image 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np 
import glob
import os
import shutil
from sklearn.cluster import KMeans 
import sys
import pickle

IMG_DIR = "/home/kishor/GWM/Meta_Cognition_Experiments/1_Clustering/Dataset/image_dataset/"
SRC_DIR = "/home/kishor/GWM/Meta_Cognition_Experiments/1_Clustering/"

def generate_feature_vector(imagelist, model):
	
	vgg16_features_list = []
	
	for idx, each_image in enumerate(image_list):
		img = image.load_img(each_image, target_size = (224, 224))
		#print("Shape 1 : {}".format(img.shape))
		img_data = image.img_to_array(img)
		#print("Shape 2 : {}".format(img_data.shape))
		img_data = np.expand_dims(img_data, axis = 0)
		#print("Shape 3 : {}".format(img_data.shape))
		
		img_data = preprocess_input(img_data)

		vgg16_features = model.predict(img_data)
		vgg16_features_np = np.array(vgg16_features)

		vgg16_features_list.append(vgg16_features_np.flatten())

	
	vgg16_features_list_np = np.array(vgg16_features_list)

	os.chdir(SRC_DIR)
	np.save("feature_vectors.npy", vgg16_features_list_np)

	return vgg16_features_list_np


if __name__ == "__main__":

	model = VGG16(weights = "imagenet", include_top = False)
	model.summary()

	'''
	img_path = "/home/kishor/GWM/Meta_Cognition_Experiments/1_Clustering/test.jpg"
	img = image.load_img(img_path, target_size = (224, 224))
	img_data = image.img_to_array(img)
	img_data = np.expand_dims(img_data, axis = 0)
	img_data = preprocess_input(img_data)

	vgg16_features = model.predict(img_data)

	print("Feature shape : {}".format(vgg16_features.shape))
	'''

	curr_dir = os.getcwd()
	os.chdir(IMG_DIR)
	image_list = glob.glob("*.jpg")
	print("Number of images : {}".format(len(image_list)))

	vector_gen = sys.argv[1]

	os.chdir(SRC_DIR)

	if (vector_gen == "yes"):
		vgg16_features_list_np = generate_feature_vector(image_list, model)
	else:
		vgg16_features_list_np = np.load("feature_vectors.npy")

	kmeans_cluster = KMeans(n_clusters = 2, random_state = 0).fit(vgg16_features_list_np)

	model_name = "image_kmeans.sav"
	pickle.dump(kmeans_cluster, open(model_name, "wb"))

	print(kmeans_cluster)

	print("Centroid : {}".format(kmeans_cluster.cluster_centers_))
	print("Labels   : {}".format(kmeans_cluster.labels_))






