import cv2
import matplotlib.pyplot as plt
import numpy as np
from focal_frequency_loss import FocalFrequencyLoss as FFL
import torchvision.transforms as transforms
def argCheck_reconstruct_image(image_path):
	import os,cv2
	if os.path.exists(image_path):
		return "image does not exists!"
	try:
		image = cv2.imread(image_path)
	except Exception as e:
		return str(e)
	return True

def reconstruct_image(image_path,width_size = 1,loss_weight = 1.0,alpha = 1.0):
	check = argCheck_reconstruct_image(image_path)
	if not check:
		return check

	def convert_to_tensor(image):
		
		transform = transforms.ToTensor()
		tensor = transform(image)
		return tensor

	image = cv2.imread(image_path)
	real = convert_to_tensor(image).unsqueeze(0)
	# find the contours from the thresholded image
	image               = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# draw all contours
	BLACK_PIXEL = 0
	WHITE_PIXEL = 255
	image = cv2.drawContours(
		image, 
		contours, 
		-1, 
		(WHITE_PIXEL,WHITE_PIXEL,WHITE_PIXEL), 
		width_size)
	# show the image with the drawn contours
	# plt.imshow(image)
	# plt.show()
	fake = convert_to_tensor(image).unsqueeze(0)
	
	reconstructed_image = image.copy()
	
	ffl = FFL(loss_weight=loss_weight, alpha=alpha)  # initialize nn.Module class
	rv = {}
	rv["image"] = image
	rv["reconstructed_image"] = reconstructed_image
	rv["focal_frequency_loss"] = ffl(fake,real).item()    
	return rv

if __name__=="__main__":
	image_path = r"C:\Users\gprak\mask.jpg"
	ans = reconstruct_image(image_path)
	print(ans)