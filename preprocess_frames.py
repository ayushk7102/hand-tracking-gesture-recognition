import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
import os

def process_handframes():
	transform = transforms.Compose([
	    transforms.ToTensor()
	])

	glob_mean = torch.Tensor([0, 0, 0])
	glob_std = torch.Tensor([0, 0, 0])
	count = 0
	for img_path in os.listdir('/home/ayush/Desktop/Stuff/DVCON/handframes'):

		x = cv.imread(os.path.join('/home/ayush/Desktop/Stuff/DVCON/handframes', img_path))
		img_tr = transform(x)
		img_np = np.array(img_tr)
		mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
	 	# print mean and std
		# print("mean and std before normalize:")
		# print("Mean of the image:", mean)
		# print("Std of the image:", std)

		glob_mean = torch.add(glob_mean, mean, alpha=1)
		glob_std = torch.add(glob_std, std, alpha=1)
		count+=1

	glob_mean,  glob_std = glob_mean/count, glob_std/count
	print('glob_mean: ', glob_mean.numpy())
	print('glob_STD: ', glob_std.numpy())



	transform_img = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize(size=[256], interpolation= transforms.functional.InterpolationMode.BILINEAR),
		transforms.CenterCrop(size=[224]),
		transforms.Normalize(glob_mean, glob_std)
		])



