'''
Generation script
	- correct physical ratios
	- run on blue crystal
'''
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
import random
import math
import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from keras.models import load_model
from scipy.stats import chisquare
import time
from keras import backend as K
import argparse
_EPSILON = K.epsilon() 

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

num_pos13_x_x = 5035390/200
num_pos13_0_0 = 2478943/200
num_neg13_x_x = 6222950/200
num_neg13_0_0 = 2829153/200 


G_pos13_x_x = load_model('/mnt/storage/scratch/am13743/MOMENTUM_GANs/4/pos_x/test_output/models/Generator_pos13_x_x_best_rchi2_mom.h5', custom_objects = {'_loss_generator':_loss_generator})
G_pos13_0_0 = load_model('/mnt/storage/scratch/am13743/MOMENTUM_GANs/1/pos_0/test_output/models/Generator_pos13_0_0_best_rchi2_mom.h5', custom_objects = {'_loss_generator':_loss_generator})
G_neg13_x_x = load_model('/mnt/storage/scratch/am13743/MOMENTUM_GANs/3/neg_x/test_output/models/Generator_neg13_x_x_best_rchi2_mom.h5', custom_objects = {'_loss_generator':_loss_generator})
G_neg13_0_0 = load_model('/mnt/storage/scratch/am13743/MOMENTUM_GANs/1/neg_0/test_output/models/Generator_neg13_0_0_best_rchi2_mom.h5', custom_objects = {'_loss_generator':_loss_generator})


means_pos_x = np.load('/mnt/storage/scratch/am13743/FULL_GAN_MIN_MAX/means_pos13_x_x.npy')
min_max_1_pos_x = np.load('/mnt/storage/scratch/am13743/FULL_GAN_MIN_MAX/min_max_pos13_x_x.npy')
min_max_2_pos_x = np.load('/mnt/storage/scratch/am13743/FULL_GAN_MIN_MAX/min_max_2_pos13_x_x.npy')

means_neg_x = np.load('/mnt/storage/scratch/am13743/FULL_GAN_MIN_MAX/means_neg13_x_x.npy')
min_max_1_neg_x = np.load('/mnt/storage/scratch/am13743/FULL_GAN_MIN_MAX/min_max_neg13_x_x.npy')
min_max_2_neg_x = np.load('/mnt/storage/scratch/am13743/FULL_GAN_MIN_MAX/min_max_2_neg13_x_x.npy')

min_max_1_pos_0 = np.load('/mnt/storage/scratch/am13743/FULL_GAN_MIN_MAX/min_max_pos13_0_0.npy')
min_max_1_neg_0 = np.load('/mnt/storage/scratch/am13743/FULL_GAN_MIN_MAX/min_max_neg13_0_0.npy')

t0 = time.time()

def normalize_6(input_array, min_max_1, min_max_2, means):

	for index in range(0, 2):
		for x in range(0, np.shape(input_array)[0]):
			input_array[x][index] = (((input_array[x][index]+0.97)/1.94)*(min_max_2[index][1] - min_max_2[index][0])+ min_max_2[index][0])
			input_array[x][index] = input_array[x][index] - means[index]
			if input_array[x][index] < 0:
				input_array[x][index] = -(input_array[x][index]**2)
			if input_array[x][index] > 0:
				input_array[x][index] = (input_array[x][index]**2)
			input_array[x][index] = input_array[x][index] + means[index]
			input_array[x][index] = (((input_array[x][index]+1)/2)*(min_max_1[index][1] - min_max_1[index][0])+ min_max_1[index][0])

	for index in range(2, 6):
		for x in range(0, np.shape(input_array)[0]):
			input_array[x][index] = (((input_array[x][index]+0.97)/1.94)*(min_max_1[index][1] - min_max_1[index][0])+ min_max_1[index][0])

	return input_array

def normalize_4(input_array, min_max_1):
	
	for index in range(0, 4):
		for x in range(0, np.shape(input_array)[0]):
			input_array[x][index] = (((input_array[x][index]+0.97)/1.94)*(min_max_1[index][1] - min_max_1[index][0])+ min_max_1[index][0])

	return input_array

def generate():
	number = np.random.poisson(num_pos13_x_x, 1)[0]
	noise = np.random.normal(0, 1, (number, 100))
	images = np.squeeze(G_pos13_x_x.predict(noise))
	pos_x_train = normalize_6(images, min_max_1_pos_x, min_max_2_pos_x, means_pos_x)
	# pos_x_train = images
	pdg = np.ones((np.shape(pos_x_train)[0],1))*13
	pos_x_train = np.concatenate((pdg, pos_x_train),axis=1)

	number = np.random.poisson(num_neg13_x_x, 1)[0]
	noise = np.random.normal(0, 1, (number, 100))
	images = np.squeeze(G_neg13_x_x.predict(noise))
	neg_x_train = normalize_6(images, min_max_1_neg_x, min_max_2_neg_x, means_neg_x)
	# neg_x_train = images
	pdg = np.ones((np.shape(neg_x_train)[0],1))*-13
	neg_x_train = np.concatenate((pdg, neg_x_train),axis=1)

	number = np.random.poisson(num_neg13_0_0, 1)[0]
	noise = np.random.normal(0, 1, (number, 100))
	images = np.squeeze(G_neg13_0_0.predict(noise))
	neg_0_train = normalize_4(images, min_max_1_neg_0)
	# neg_0_train = images
	two_zeros = np.zeros((np.shape(neg_0_train)[0],2))
	neg_0_train = np.concatenate((two_zeros, neg_0_train),axis=1)
	pdg = np.ones((np.shape(neg_0_train)[0],1))*-13
	neg_0_train = np.concatenate((pdg, neg_0_train),axis=1)

	number = np.random.poisson(num_pos13_0_0, 1)[0]
	noise = np.random.normal(0, 1, (number, 100))
	images = np.squeeze(G_pos13_0_0.predict(noise))
	pos_0_train = normalize_4(images, min_max_1_pos_0)
	# pos_0_train = images
	two_zeros = np.zeros((np.shape(pos_0_train)[0],2))
	pos_0_train = np.concatenate((two_zeros, pos_0_train),axis=1)
	pdg = np.ones((np.shape(pos_0_train)[0],1))*13
	pos_0_train = np.concatenate((pdg, pos_0_train),axis=1)

	generated_training = np.concatenate((pos_x_train, neg_x_train, neg_0_train, pos_0_train), axis=0)

	return generated_training

complete_saved = 0

for x in range(0, 17):
	saved_muons = 0
	first_save = True

	while saved_muons < 1000000:

		if first_save == False:
			total_saved = np.load('/mnt/storage/scratch/am13743/GAN_GEN_MUONS_FULL/full_muons_%d.npy'%x)
		else:
			total_saved = np.empty((0, 7))

		generated_training = generate()

		total_saved = np.concatenate((total_saved, generated_training), axis=0)

		np.save('/mnt/storage/scratch/am13743/GAN_GEN_MUONS_FULL/full_muons_%d.npy'%x, total_saved)
		
		first_save = False

		saved_muons += np.shape(generated_training)[0]
		
	complete_saved += saved_muons


t1 = time.time()
total_n = t1-t0
print('Generated:', complete_saved, 'in %.1fs'%total_n)




