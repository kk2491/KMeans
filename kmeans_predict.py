from sklearn.cluster import KMeans
import pickle
import sys
from keras.preprocessing import image 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np 


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

	return vgg16_features_list_np


if __name__ == "__main__":

	model = VGG16(weights = "imagenet", include_top = False)
	model.summary()

	test_data = sys.argv[1]

	img = image.load_img(test_data, target_size = (224, 224))
	img_data = image.img_to_array(img)
	img_data = np.expand_dims(img_data, axis = 0)
	img_data = preprocess_input(img_data)

	vgg16_features = model.predict(img_data)
	vgg16_features_np = np.array(vgg16_features)
	vgg16_features_np = vgg16_features_np.flatten()

	#vgg16_features_list.append(vgg16_features_np.flatten())

	#image_list = [test_data]
	#vgg16_features_list_np = generate_feature_vector(image_list, model)

	#vgg16_features = model.predict(vgg16_features_np)

	kmeans_model = pickle.load(open("image_kmeans.sav", "rb"))

	print(kmeans_model)

	predicted_group = kmeans_model.predict(img_data)

	print("Prediction : {}".format(predicted_group))