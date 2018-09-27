'''
Plot 4 way BDT plots for defined generators.
Plots BDTs based on both:
	[x,y,z,p_x,p_y,p_z]
	[z,p,p_t,p_x,p_y,p_z]
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
from keras.models import load_model
from scipy.stats import chisquare
import os
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
from keras import backend as K
import argparse
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from matplotlib.colors import LogNorm

_EPSILON = K.epsilon() 

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

# ('Saved', 2238402, 'neg13 0 0 muons.')
# ('Saved', 2222657, 'pos13 0 0 muons.')
# ('Saved', 1798199, 'neg13 x x muons.')
# ('Saved', 1777509, 'pos13 x x muons.')

f = 3
num_pos13_x_x = 1777509/100 * f
num_pos13_0_0 = 2222657/100 * f
num_neg13_x_x = 1798199/100 * f
num_neg13_0_0 = 2238402/100 * f

# G_pos13_x_x = load_model('Generator_pos13_x_x_best_rchi2_best_every_10_78_mom.h5', custom_objects = {'_loss_generator':_loss_generator})
# G_pos13_x_x = load_model('Generator_pos13_x_x_best_rchi2_best_every_10_62_mom.h5', custom_objects = {'_loss_generator':_loss_generator})
G_pos13_x_x = load_model('Generator_pos13_x_x_best_rchi2_best_every_10_70_mom.h5', custom_objects = {'_loss_generator':_loss_generator})
G_neg13_x_x = load_model('Generator_neg13_x_x_best_rchi2_best_every_10_111_mom.h5', custom_objects = {'_loss_generator':_loss_generator})
G_pos13_0_0 = load_model('Generator_pos13_0_0_best_rchi2_best_every_10_59_mom.h5', custom_objects = {'_loss_generator':_loss_generator})
G_neg13_0_0 = load_model('Generator_neg13_0_0_best_rchi2_best_every_10_94_mom.h5', custom_objects = {'_loss_generator':_loss_generator})


means_pos_x = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/means_pos13_x_x.npy')
min_max_1_pos_x = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/min_max_pos13_x_x.npy')
min_max_2_pos_x = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/min_max_2_pos13_x_x.npy')

means_neg_x = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/means_neg13_x_x.npy')
min_max_1_neg_x = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/min_max_neg13_x_x.npy')
min_max_2_neg_x = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/min_max_2_neg13_x_x.npy')

min_max_1_pos_0 = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/min_max_pos13_0_0.npy')
min_max_1_neg_0 = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/min_max_neg13_0_0.npy')

def normalize_6(input_array, min_max_1, min_max_2, means):

	# print(min_max_1, min_max_2, means)

	# print(min_max_1)

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

def plot_bdt(images, which, num):



	if which == 'pos13_x_x':
		X_train = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/5_pos13_x_x_second_norm.npy')
	if which == 'neg13_x_x':
		X_train = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/1_neg13_x_x_second_norm.npy')
	if which == 'neg13_0_0':
		X_train = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/1_save_neg13_0_0_first_norm.npy')
	if which == 'pos13_0_0':
		X_train = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/1_save_pos13_0_0_first_norm.npy')

	bdt_train_size = 50000

	muon_weights, X_train = np.split(X_train, [1], axis=1)
	list_for_np_choice = np.arange(np.shape(X_train)[0]) 
	muon_weights = np.squeeze(muon_weights/np.sum(muon_weights))	
	# print(np.shape(list_for_np_choice), np.shape(muon_weights))
	random_indicies = np.random.choice(list_for_np_choice, size=(2,50000), p=muon_weights, replace=False)

	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

	real_training_data = X_train[random_indicies[0]]

	real_test_data = X_train[random_indicies[1]]

	fake_training_data = np.squeeze(images[:bdt_train_size])

	fake_test_data = np.squeeze(images[bdt_train_size:])

	real_training_labels = np.ones(bdt_train_size)

	fake_training_labels = np.zeros(bdt_train_size)

	total_training_data = np.concatenate((real_training_data, fake_training_data))

	total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

	clf.fit(total_training_data, total_training_labels)

	out_real = clf.predict_proba(real_test_data)

	out_fake = clf.predict_proba(fake_test_data)

	plt.subplot(2,2,num)
	plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['Real','Generated'], histtype='step', linewidth='2', color=['#4772FF','#FF5959'], range=[0,1])
	x_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	plt.xticks()
	plt.tick_params(axis='y', which='both', labelsize=7)

	plt.xlabel('BDT Response', fontsize='x-small')
	if num == 2:
		plt.legend(loc='upper right', fontsize='small')
	# plt.savefig('plots/BDT_out_%s.png'%(which), bbox_inches='tight')
	# plt.close('all')
##############################
plot_bdt_bool = True


plt.figure(figsize=(8,8))
number = np.random.poisson(num_pos13_x_x, 1)[0]
if plot_bdt_bool == True:
	number = 100000
noise = np.random.normal(0, 1, (number, 100))
images = np.squeeze(G_pos13_x_x.predict(noise))
if plot_bdt_bool == True:
	plot_bdt(images, 'pos13_x_x',1)
# quit()
# pos_x_train = normalize_6(images, min_max_1_pos_x, min_max_2_pos_x, means_pos_x)
# pdg = np.ones((np.shape(pos_x_train)[0],1))*13
# pos_x_train = np.concatenate((pdg, pos_x_train),axis=1)

number = np.random.poisson(num_neg13_x_x, 1)[0]
if plot_bdt_bool == True:
	number = 100000
noise = np.random.normal(0, 1, (number, 100))
images = np.squeeze(G_neg13_x_x.predict(noise))
if plot_bdt_bool == True:
	plot_bdt(images, 'neg13_x_x',2)


# neg_x_train = normalize_6(images, min_max_1_neg_x, min_max_2_neg_x, means_neg_x)
# pdg = np.ones((np.shape(neg_x_train)[0],1))*-13
# neg_x_train = np.concatenate((pdg, neg_x_train),axis=1)

number = np.random.poisson(num_neg13_0_0, 1)[0]
if plot_bdt_bool == True:
	number = 100000
noise = np.random.normal(0, 1, (number, 100))
images = np.squeeze(G_neg13_0_0.predict(noise))
if plot_bdt_bool == True:
	plot_bdt(images, 'neg13_0_0',3)
# neg_0_train = normalize_4(images, min_max_1_neg_0)
# two_zeros = np.zeros((np.shape(neg_0_train)[0],2))
# neg_0_train = np.concatenate((two_zeros, neg_0_train),axis=1)
# pdg = np.ones((np.shape(neg_0_train)[0],1))*-13
# neg_0_train = np.concatenate((pdg, neg_0_train),axis=1)

number = np.random.poisson(num_pos13_0_0, 1)[0]
if plot_bdt_bool == True:
	number = 100000
noise = np.random.normal(0, 1, (number, 100))
images = np.squeeze(G_pos13_0_0.predict(noise))
if plot_bdt_bool == True:
	plot_bdt(images, 'pos13_0_0',4)
# pos_0_train = normalize_4(images, min_max_1_pos_0)
# two_zeros = np.zeros((np.shape(pos_0_train)[0],2))
# pos_0_train = np.concatenate((two_zeros, pos_0_train),axis=1)
# pdg = np.ones((np.shape(pos_0_train)[0],1))*13
# pos_0_train = np.concatenate((pdg, pos_0_train),axis=1)
if plot_bdt_bool == True:
	plt.savefig('plots/BDT_out_regular.png', bbox_inches='tight')
##############################



def plot_bdt_mom(images, which, num, dimensions):

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


	if which == 'pos13_x_x':
		X_train = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/5_pos13_x_x_second_norm.npy')
		charge, X_train = np.split(X_train,[1],axis=1)
	if which == 'neg13_x_x':
		X_train = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/1_neg13_x_x_second_norm.npy')
		charge, X_train = np.split(X_train,[1],axis=1)
	if which == 'neg13_0_0':
		X_train = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/1_save_neg13_0_0_first_norm.npy')
		charge, X_train = np.split(X_train,[1],axis=1)
	if which == 'pos13_0_0':
		X_train = np.load('/Users/am13743/Desktop/GANs/FULL_THOMAS_DATA/1_save_pos13_0_0_first_norm.npy')
		charge, X_train = np.split(X_train,[1],axis=1)

	print(np.shape(X_train), np.shape(images))
	print(images)

	if which == 'neg13_0_0':
		images_squeeze = normalize_4(np.squeeze(images), min_max_1_neg_0)
		plot_data_squeeze = normalize_4(np.squeeze(X_train), min_max_1_neg_0)
	if which == 'neg13_x_x':
		images_squeeze = normalize_6(np.squeeze(images), min_max_1_neg_x, min_max_2_neg_x, means_neg_x)
		plot_data_squeeze = normalize_6(np.squeeze(X_train), min_max_1_neg_x, min_max_2_neg_x, means_neg_x)
	if which == 'pos13_0_0':
		images_squeeze = normalize_4(np.squeeze(images), min_max_1_pos_0)
		plot_data_squeeze = normalize_4(np.squeeze(X_train), min_max_1_pos_0)
	if which == 'pos13_x_x':
		images_squeeze = normalize_6(np.squeeze(images), min_max_1_pos_x, min_max_2_pos_x, means_pos_x)
		plot_data_squeeze = normalize_6(np.squeeze(X_train), min_max_1_pos_x, min_max_2_pos_x, means_pos_x)

	print(np.shape(plot_data_squeeze), np.shape(images_squeeze))
	print(images_squeeze)

	if dimensions == 6:
		pos_xy, z_real, mom_array = np.split(plot_data_squeeze,[2,3],axis=1)
	if dimensions == 4:
		z_real, mom_array = np.split(plot_data_squeeze,[1],axis=1)
	p_real = mom_array.sum(axis=1)
	p_x_real, p_y_real, p_z_real = np.split(mom_array, [1,2], axis=1)
	p_x_sq_real = np.multiply(p_x_real, p_x_real)
	p_y_sq_real = np.multiply(p_y_real, p_y_real)
	sqs_array_real = np.concatenate((p_x_sq_real,p_y_sq_real),axis=1)
	sum_sqs_real = sqs_array_real.sum(axis=1)
	p_t_real = np.sqrt(sum_sqs_real)
	p_real = np.expand_dims(p_real,1)
	p_t_real = np.expand_dims(p_t_real,1)

	real_mom = np.concatenate((z_real, p_x_real, p_y_real, p_real, p_t_real),axis=1)
	real_mom_train = real_mom[:50000]
	real_mom_test = real_mom[50000:100000]

		
	if dimensions == 6:
		pos_xy, z_fake, mom_array = np.split(images_squeeze,[2,3],axis=1)
	if dimensions == 4:
		z_fake, mom_array = np.split(images_squeeze,[1],axis=1)
	p_fake = mom_array.sum(axis=1)
	p_x_fake, p_y_fake, p_z_fake = np.split(mom_array, [1,2], axis=1)
	p_x_sq_fake = np.multiply(p_x_fake, p_x_fake)
	p_y_sq_fake = np.multiply(p_y_fake, p_y_fake)
	sqs_array_fake = np.concatenate((p_x_sq_fake,p_y_sq_fake),axis=1)
	sum_sqs_fake = sqs_array_fake.sum(axis=1)
	p_t_fake = np.sqrt(sum_sqs_fake)
	p_fake = np.expand_dims(p_fake,1)
	p_t_fake = np.expand_dims(p_t_fake,1)

	fake_mom = np.concatenate((z_fake, p_x_fake, p_y_fake, p_fake, p_t_fake),axis=1)
	fake_mom_train = fake_mom[:50000]
	fake_mom_test = fake_mom[50000:100000]



	clf_mom = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

	bdt_train_size = 50000

	real_training_labels = np.ones(bdt_train_size)

	fake_training_labels = np.zeros(bdt_train_size)

	total_training_data = np.concatenate((real_mom_train, fake_mom_train))

	total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

	clf_mom.fit(total_training_data, total_training_labels)

	out_real = clf_mom.predict_proba(real_mom_test)

	out_fake = clf_mom.predict_proba(fake_mom_test)


	# plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
	# plt.xlabel('Output of BDT')
	# plt.legend(loc='upper right')
	# if blue_crystal == True:
	# 	plt.savefig('/mnt/storage/scratch/am13743/%s/test_output/bdt_mom/BDT_out_%d.png'%(file_number,step), bbox_inches='tight')
	# else:
	# 	plt.savefig('test_output/bdt_mom/BDT_out_%d.png'%(step), bbox_inches='tight')
	# plt.close('all')


	plt.subplot(2,2,num)
	plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['Real','Generated'], histtype='step', linewidth='2', color=['#4772FF','#FF5959'], range=[0,1])
	x_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	plt.xticks()
	plt.tick_params(axis='y', which='both', labelsize=7)

	plt.xlabel('BDT Response', fontsize='x-small')
	if num == 2:
		plt.legend(loc='upper right', fontsize='small')
	# plt.savefig('plots/BDT_out_%s.png'%(which), bbox_inches='tight')
	# plt.close('all')






plot_bdt_bool = True


plt.figure(figsize=(8,8))
number = np.random.poisson(num_pos13_x_x, 1)[0]
if plot_bdt_bool == True:
	number = 100000
noise = np.random.normal(0, 1, (number, 100))
images = np.squeeze(G_pos13_x_x.predict(noise))
if plot_bdt_bool == True:
	plot_bdt_mom(images, 'pos13_x_x',1,6)
# quit()
# pos_x_train = normalize_6(images, min_max_1_pos_x, min_max_2_pos_x, means_pos_x)
# pdg = np.ones((np.shape(pos_x_train)[0],1))*13
# pos_x_train = np.concatenate((pdg, pos_x_train),axis=1)

number = np.random.poisson(num_neg13_x_x, 1)[0]
if plot_bdt_bool == True:
	number = 100000
noise = np.random.normal(0, 1, (number, 100))
images = np.squeeze(G_neg13_x_x.predict(noise))
if plot_bdt_bool == True:
	plot_bdt_mom(images, 'neg13_x_x',2,6)


# neg_x_train = normalize_6(images, min_max_1_neg_x, min_max_2_neg_x, means_neg_x)
# pdg = np.ones((np.shape(neg_x_train)[0],1))*-13
# neg_x_train = np.concatenate((pdg, neg_x_train),axis=1)

number = np.random.poisson(num_neg13_0_0, 1)[0]
if plot_bdt_bool == True:
	number = 100000
noise = np.random.normal(0, 1, (number, 100))
images = np.squeeze(G_neg13_0_0.predict(noise))
if plot_bdt_bool == True:
	plot_bdt_mom(images, 'neg13_0_0',3,4)
# neg_0_train = normalize_4(images, min_max_1_neg_0)
# two_zeros = np.zeros((np.shape(neg_0_train)[0],2))
# neg_0_train = np.concatenate((two_zeros, neg_0_train),axis=1)
# pdg = np.ones((np.shape(neg_0_train)[0],1))*-13
# neg_0_train = np.concatenate((pdg, neg_0_train),axis=1)

number = np.random.poisson(num_pos13_0_0, 1)[0]
if plot_bdt_bool == True:
	number = 100000
noise = np.random.normal(0, 1, (number, 100))
images = np.squeeze(G_pos13_0_0.predict(noise))
if plot_bdt_bool == True:
	plot_bdt_mom(images, 'pos13_0_0',4,4)
# pos_0_train = normalize_4(images, min_max_1_pos_0)
# two_zeros = np.zeros((np.shape(pos_0_train)[0],2))
# pos_0_train = np.concatenate((two_zeros, pos_0_train),axis=1)
# pdg = np.ones((np.shape(pos_0_train)[0],1))*13
# pos_0_train = np.concatenate((pdg, pos_0_train),axis=1)
if plot_bdt_bool == True:
	plt.savefig('plots/BDT_out_mom.png', bbox_inches='tight')

quit()








