import glob
import shutil
import os
import random

CAT_RAW_DIR = "/home/kishor/GWM/Meta_Cognition_Experiments/1_Clustering/Dataset/raw_dataset/PetImages/Cat/"
DOG_RAW_DIR = "/home/kishor/GWM/Meta_Cognition_Experiments/1_Clustering/Dataset/raw_dataset/PetImages/Dog/"
DEST_DIR = "/home/kishor/GWM/Meta_Cognition_Experiments/1_Clustering/Dataset/image_dataset"

def random_selection(path):

	os.chdir(path)
	print(os.getcwd())
	images = glob.glob("*.jpg")
	print(len(images))

	random_1000 = random.sample(images, k = 1000)
	print(len(random_1000))

	return random_1000

# rename the file and copy to dataset directory
def copy_images(image_list, tag):

	for each_image in image_list:

		if tag == "CAT":
			new_filename = DEST_DIR+"/cat_"+each_image
		elif tag == "DOG":
			new_filename = DEST_DIR+"/dog_"+each_image
		else:
			print("Nothing here")

		shutil.copy(each_image, new_filename)
		

	return


if __name__ == "__main__":
	
	curr_dir = os.getcwd()

	random_cat_1000 = random_selection(CAT_RAW_DIR)
	cat_copy_status = copy_images(random_cat_1000, "CAT")

	random_dog_1000 = random_selection(DOG_RAW_DIR)
	#cat_copy_status = copy_images(random_cat_1000, "CAT")
	dog_copy_status = copy_images(random_dog_1000, "DOG")

