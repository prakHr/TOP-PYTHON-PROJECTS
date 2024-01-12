### Example script showing how to perform a Total-Variation filtering with proxTV
import prox_tv as ptv
import numpy as np
import matplotlib.pyplot as plt
import time
import skimage as ski
from skimage import io, color, util
import os

def clear_noise_from_signal(noisy_signal,lam = 20,algorithm = "tv1_1d"):
	start = time.time()
	cleaned_signal = eval(f"ptv.{algorithm}(noisy_signal,lam)")
	end = time.time()
	print('Elapsed time '+str(end-start))
	return {
		"noisy":noisy_signal,
		"cleaned":cleaned_signal
	}

def clear_noise_from_image(noisy_image,lam = 20,algorithm = "tv1_1d"):
	X = noisy_image
	X = ski.img_as_float(X)
	X = color.rgb2gray(X)
	start = time.time()
	cleaned_image = eval(f"ptv.{algorithm}(noisy_image,lam)")
	end = time.time()
	print('Elapsed time '+str(end-start))
	return {
		"noisy":noisy_image,
		"cleaned":cleaned_image
	}


def multiple_cleaning_signal_or_image(noisy_signal_or_noisy_image,lam_list,algorithm_list):

	rv = {}
	for lam,algorithm in zip(lam_list,algorithm_list):
		rv = clear_noise_from_signal(noisy_signal_or_noisy_image,lam,algorithm)
		noisy_signal_or_noisy_image = rv["cleaned"]
	return rv

if __name__=="__main__":
	# Generate sinusoidal signal
	N = 1000
	s = np.sin(np.arange(1,N+1)/10.0) + np.sin(np.arange(1,N+1)/100.0)
	n = s + 0.5*np.random.randn(*np.shape(s))
	lam=100;
	ans = clear_noise_from_signal(n,lam,"tv2_1d")
	print(ans)

	path = "colors.png"
	X = io.imread(path)
	X = ski.img_as_float(X)
	X = color.rgb2gray(X)
	# Introduce noise
	noiseLevel = 0.01
	N = util.random_noise(X, mode='speckle', var=noiseLevel)
	ans = clear_noise_from_image(N,0.15,"tv1_2d")
	print(ans)
	
