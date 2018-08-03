# -*- coding: utf-8 -*-
""" Simple implementation of Generative Adversarial Neural Network """

import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LocallyConnected2D, Conv2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam, RMSpro
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

file_number = 0

blue_crystal = False

_EPSILON = K.epsilon() # 10^-7 by default. Epsilon is used as a small constant to avoid ever dividing by zero. 

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)
# This is the traditional loss function for GAN generator.
# In this GAN real images are labeled 1 and generated images 0. 
# This loss function returns high then when more images are label 0 (spotted by D as fakes).

class GAN(object):

	def __init__(self, width=1, height=6, channels=1):

		parser = argparse.ArgumentParser()

		# Default values currently set as best discovered combination of hyper parameters.

		parser.add_argument('-l', action='store', dest='learning_rate', type=float,
		                    help='learning rate', default=0.00005)

		parser.add_argument('-o', action='store', dest='optimizer_choice', type=int,
							default = 0,
		                    help='0 - Adam with AMSgrad, 1 - Adam, 2 - RMSprop')

		parser.add_argument('-nG', action='store', default=6, type=float,
		                    dest='node_factor_G',
		                    help='Node factor')

		parser.add_argument('-nD', action='store', default=3, type=float,
		                    dest='node_factor_D',
		                    help='Node factor')

		parser.add_argument('-layersG', action='store', default=2, type=int,
		                    dest='num_layersG',
		                    help='The number of layers')

		parser.add_argument('-layersD', action='store', default=2, type=int,
		                    dest='num_layersD',
		                    help='The number of layers')

		results = parser.parse_args()

		self.num_layersG = results.num_layersG
		self.num_layersD = results.num_layersD
		self.width = width
		self.height = height
		self.channels = channels
		self.node_factor_G = int(results.node_factor_G)
		self.node_factor_D = int(results.node_factor_D)

		self.shape = (self.width, self.height, self.channels) # Define shape of muon 'images'

		if int(results.optimizer_choice) == 0:
			print('Adam with amsgrad')
			self.optimizerG = Adam(lr=results.learning_rate, beta_1=0.5, decay=5e-6, amsgrad=True)
			self.optimizerD = Adam(lr=results.learning_rate, beta_1=0.5, decay=5e-6, amsgrad=True)

		if int(results.optimizer_choice) == 1:
			print('Adam without amsgrad')
			self.optimizerG = Adam(lr=results.learning_rate, beta_1=0.5, decay=8e-8)
			self.optimizerD = Adam(lr=results.learning_rate, beta_1=0.5, decay=8e-8)

		if int(results.optimizer_choice) == 2:
			print('RMSprop')
			self.optimizerG = RMSprop(lr=results.learning_rate, rho=0.9, epsilon=None, decay=0.0)
			self.optimizerD = RMSprop(lr=results.learning_rate, rho=0.9, epsilon=None, decay=0.0)
		
		self.G = self.__generator()
		self.G.compile(loss=_loss_generator, optimizer=self.optimizerG)

		self.D = self.__discriminator()
		self.D.compile(loss='binary_crossentropy', optimizer=self.optimizerD, metrics=['accuracy'])

		# Create a stacked model of G and D. In a GAN, G is never trained directly.
		# The training of G requires feedback from D. 
		# It is this stacked_generator_discriminator is trained.
		self.stacked_generator_discriminator = self.__stacked_generator_discriminator()
		self.stacked_generator_discriminator.compile(loss=_loss_generator, optimizer=self.optimizerD)


	def __generator(self):

		model = Sequential()

		model.add(Dense(int(256*self.node_factor_G), input_shape=(100,)))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))

		if self.num_layersG >= 2:
			print('G_2')
			model.add(Dense(int(512*self.node_factor_G)))
			model.add(LeakyReLU(alpha=0.2))
			model.add(BatchNormalization(momentum=0.8))
		if self.num_layersG >= 3:
			print('G_3')
			model.add(Dense(int(1024*self.node_factor_G)))
			model.add(Dropout(0.25))
			model.add(LeakyReLU(alpha=0.2))
			model.add(BatchNormalization(momentum=0.8))
		if self.num_layersG >= 4:
			print('G_4')
			model.add(Dense(int(2048*self.node_factor_G)))
			model.add(Dropout(0.25))
			model.add(LeakyReLU(alpha=0.2))
			model.add(BatchNormalization(momentum=0.8))
		if self.num_layersG >= 5:
			print('G_5')
			model.add(Dense(int(2048*2*self.node_factor_G)))
			model.add(Dropout(0.25))
			model.add(LeakyReLU(alpha=0.2))
			model.add(BatchNormalization(momentum=0.8))
		if self.num_layersG >= 6:
			print('G_6')
			model.add(Dense(int(2048*4*self.node_factor_G)))
			model.add(Dropout(0.25))
			model.add(LeakyReLU(alpha=0.2))
			model.add(BatchNormalization(momentum=0.8))

		model.add(Dense(self.width  * self.height * self.channels, activation='tanh'))
		model.add(Reshape((self.width, self.height, self.channels)))

		model.summary()

		return model

	def __discriminator(self):

		model = Sequential()

		model.add(Flatten(input_shape=self.shape))
		model.add(Dense((int(256 * self.node_factor_D)), input_shape=self.shape))
		model.add(LeakyReLU(alpha=0.2))

		if self.num_layersD >= 2:
			print('D_2')
			model.add(Dense((int(512 * self.node_factor_D))))
			model.add(LeakyReLU(alpha=0.2))
			model.add(Dropout(0.25))
		if self.num_layersD >= 3:
			print('D_3')
			model.add(Dense((int(1024 * self.node_factor_D))))
			model.add(LeakyReLU(alpha=0.2))
			model.add(Dropout(0.25))
		if self.num_layersD >= 4:
			print('D_4')
			model.add(Dense((int(2048 * self.node_factor_D))))
			model.add(LeakyReLU(alpha=0.2))
			model.add(Dropout(0.25))
		if self.num_layersD >= 5:
			print('D_5')
			model.add(Dense((int(2048 * 2 * self.node_factor_D))))
			model.add(LeakyReLU(alpha=0.2))
			model.add(Dropout(0.25))
		if self.num_layersD >= 6:
			print('D_6')
			model.add(Dense((int(2048 * 4 * self.node_factor_D))))
			model.add(LeakyReLU(alpha=0.2))
			model.add(Dropout(0.25))

		model.add(Dense(1, activation='sigmoid'))

		model.summary()

		return model

	def __stacked_generator_discriminator(self):

		self.D.trainable = False

		model = Sequential()
		model.add(self.G)
		model.add(self.D)

		return model

	def train(self, X_train, muon_weights, epochs=10000000, batch = 100, save_interval = 5000):

		# Initialize arrays to store progress information, losses and rchi2 values. 

		d_loss_list = np.empty((0,2))

		g_loss_list = np.empty((0,2))

		bdt_rchi2_list = np.empty((0,2))

		bdt_sum_overlap_list = np.empty((0,2))

		t0 = time.time()

		# Start best_bdt_rchi2 value extremely high to be sure early values are lower.

		best_bdt_rchi2 = 1E30

		list_for_np_choice = np.arange(np.shape(X_train)[0]) 
		# Simple list of ordered integers that will be fed to np.random.choice (only takes 1D array)
		# This list of ints is alligned with muon_weights array.
		# This allows weight corrected sampling.

		for cnt in range(epochs):

			# Previous un-weighted sampling.
			# random_index = np.random.randint(0, len(X_train) - batch/2)
			# legit_images = X_train[random_index : random_index + int(batch/2)].reshape(int(batch/2), self.width, self.height, self.channels)

			# Weighted sampling.
			random_indicies = np.random.choice(list_for_np_choice, size=50, p=muon_weights, replace=False)
			legit_images = X_train[random_indicies].reshape(int(batch/2), self.width, self.height, self.channels)

			for x in range(0, np.shape(legit_images)[0]):# Add small noise to all input, need to check is this is helping.
				for index in range(0, 6):
					noise = np.random.normal(0,0.001)
					if legit_images[x,0,index,0] + noise > 0:
						legit_images[x,0,index,0] = legit_images[x,0,index,0] + noise
					else:
						legit_images[x,0,index,0] = legit_images[x,0,index,0]

			# Generate sample of fake images. Initially when G knows nothing this will just be random noise.
			gen_noise = np.random.normal(0, 1, (int(batch/2), 100))
			syntetic_images = self.G.predict(gen_noise)

			# Label real images with 1, and fake images with 0.
			legit_labels = np.ones((int(batch/2), 1))
			gen_labels = np.zeros((int(batch/2), 1))

			# Add normal noise to labels, need to understand if this helps reduce chance of mode collapse.
			for i in range(0, len(legit_labels)):
				legit_labels[i] = legit_labels[i] + np.random.normal(0,0.3)
			for i in range(0, len(gen_labels)):
				gen_labels[i] = gen_labels[i] + np.random.normal(0,0.3)
			
			d_loss_legit = self.D.train_on_batch(legit_images, legit_labels)
			d_loss_gen = self.D.train_on_batch(syntetic_images, gen_labels)

			# Feed latent noise to stacked_generator_discriminator for training. Misslabel it (unsure if this affects anything). 
			noise = np.random.normal(0, 1, (batch, 100))
			y_mislabled = np.ones((batch, 1))

			g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

			if cnt % 100 == 0: print ('epoch: %d, [Discriminator :: d_loss: %f %f], [ Generator :: loss: %f]' % (cnt, d_loss_legit[0], d_loss_gen[0], g_loss))

			d_loss_list = np.append(d_loss_list, [[cnt,(d_loss_legit[0]+d_loss_gen[0])/2]], axis=0)
			g_loss_list = np.append(g_loss_list, [[cnt, g_loss]], axis=0)


			if cnt > 1 and cnt % save_interval == 0:

				plt.subplot(1,2,1)
				plt.title('d loss')
				plt.plot(d_loss_list[:,0],d_loss_list[:,1])
				plt.subplot(1,2,2)
				plt.title('g loss')
				plt.plot(g_loss_list[:,0],g_loss_list[:,1])
				if blue_crystal == True:
					plt.savefig('/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/loss.png'%file_number)
				else:
					plt.savefig('test_output/loss.png')
				plt.close('all')

				bdt_rchi2_list, bdt_sum_overlap_list = self.plot_images(bdt_rchi2_list, t0, bdt_sum_overlap_list, list_for_np_choice, muon_weights, save2file=True, step=cnt)

				if bdt_rchi2_list[-1][1] < best_bdt_rchi2:
					print('Saving best.')
					if blue_crystal == True:
						with open("/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/models/best_bdt_rchi2.txt"%file_number, "a") as myfile:
							myfile.write('\n %d, %.3f %.3f'%(cnt, bdt_rchi2_list[-1][1], bdt_sum_overlap_list[-1][1]))
					else:
						with open("test_output/models/best_bdt_rchi2.txt", "a") as myfile:
							myfile.write('\n %d, %.3f %.3f'%(cnt, bdt_rchi2_list[-1][1], bdt_sum_overlap_list[-1][1]))
					if blue_crystal == True:
						self.G.save('/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/models/Generator_neg13_x_x.h5'%file_number)
					else:
						self.G.save('test_output/models/Generator_neg13_x_x.h5')
					best_bdt_rchi2 = bdt_rchi2_list[-1][1]

	def plot_images(self, bdt_rchi2_list, t0, bdt_sum_overlap_list, list_for_np_choice, muon_weights, save2file=False, samples=16, step=0):

		noise_size = 100000

		bdt_train_size = 50000

		noise = np.random.normal(0, 1, (noise_size, 100))

		images = self.G.predict(noise)

		# Initialise BDT classifier used to check progress.
		clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

		# Create training samples for BDT.
		random_indicies = np.random.choice(list_for_np_choice, size=(2,50000), p=muon_weights, replace=False)

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


		plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
		plt.legend(loc='upper right')
		if blue_crystal == True:
			plt.savefig('/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/bdt/BDT_out_%d.png'%(file_number,step), bbox_inches='tight')
		else:
			plt.savefig('test_output/bdt/BDT_out_%d.png'%(step), bbox_inches='tight')
		plt.close('all')

		# Create histograms from the output of BDT.
		real_hist = np.histogram(out_real[:,1],bins=100,range=[0,1])
		fake_hist = np.histogram(out_fake[:,1],bins=100,range=[0,1])

		# Compute rChi2 and overlap values for these histogram distrbiutions.

		bdt_chi2 = 0
		dof = 0

		for bin_index in range(0, 100):
			if real_hist[0][bin_index] > 9 and fake_hist[0][bin_index] > 9:
				dof += 1
				bdt_chi2 += ((real_hist[0][bin_index] - fake_hist[0][bin_index])**2)/((math.sqrt((math.sqrt(real_hist[0][bin_index]))**2+(math.sqrt(fake_hist[0][bin_index]))**2))**2)
		if dof > 0:
			bdt_reduced_chi2 = bdt_chi2/dof
		else:
			bdt_reduced_chi2 = 1E8

		bdt_rchi2_list = np.append(bdt_rchi2_list, [[step, bdt_reduced_chi2]], axis=0)

		plt.plot(bdt_rchi2_list[:,0], bdt_rchi2_list[:,1])
		plt.yscale('log', nonposy='clip')
		if blue_crystal == True:
			plt.savefig('/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/bdt_rchi2.png'%file_number, bbox_inches='tight')
		else:
			plt.savefig('test_output/bdt_rchi2.png', bbox_inches='tight')
		plt.close('all')

		sum_overlap = 0

		for bin_index in range(0, 100):
			if real_hist[0][bin_index] < fake_hist[0][bin_index]:
				sum_overlap += real_hist[0][bin_index]
			elif fake_hist[0][bin_index] < real_hist[0][bin_index]:
				sum_overlap += fake_hist[0][bin_index]
			else:
				sum_overlap += fake_hist[0][bin_index]

		bdt_sum_overlap_list = np.append(bdt_sum_overlap_list, [[step, sum_overlap]], axis=0)

		plt.plot(bdt_sum_overlap_list[:,0], bdt_sum_overlap_list[:,1])
		plt.yscale('log', nonposy='clip')
		if blue_crystal == True:
			plt.savefig('/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/bdt_overlap.png'%file_number, bbox_inches='tight')
		else:
			plt.savefig('test_output/bdt_overlap.png', bbox_inches='tight')
		plt.close('all')

		def compare_cross_hists_2d_hist(index_1, index_2):
			plt.subplot(1,2,1)
			plt.title('Real Properties')
			plt.hist2d(X_train[:100000,index_1],X_train[:100000,index_2], bins = 100, norm=LogNorm(), range=[[-1,1],[-1,1]])
			plt.subplot(1,2,2)
			plt.title('Generated blur')
			plt.hist2d(images[:,0,index_1,0],images[:,0,index_2,0], bins = 100, norm=LogNorm(), range=[[-1,1],[-1,1]])
			if blue_crystal == True:
				plt.savefig('/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/compare_cross_blur_%d_%d_2D.png'%(file_number,index_1, index_2), bbox_inches='tight')
			else:
				plt.savefig('test_output/compare_cross_blur_%d_%d_2D.png'%(index_1, index_2), bbox_inches='tight')
			plt.close('all')

		for x in range(0,6):
			for y in range(x+1,6):
				compare_cross_hists_2d_hist(x,y)

		for index in range(0, 6):
			plt.hist([X_train[:100000,index],images[:,0,index,0]], bins = 250,label=['real','gen'], histtype='step')
			plt.legend(loc='upper right')
			if blue_crystal == True:
				plt.savefig('/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/%d/%d.png'%(file_number,index,step), bbox_inches='tight')
			else:
				plt.savefig('test_output/%d/%d.png'%(index,step), bbox_inches='tight')
			plt.close('all')

		for index in range(0, 6):
			plt.hist([X_train[:100000,index],images[:,0,index,0]], bins = 250,label=['real','gen'], histtype='step')
			plt.legend(loc='upper right')
			if blue_crystal == True:
				plt.savefig('/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/current_%d.png'%(file_number,index), bbox_inches='tight')
			else:
				plt.savefig('test_output/current_%d.png'%(index), bbox_inches='tight')
			plt.close('all')

		t1 = time.time()
		time_so_far = t1-t0
		if blue_crystal == True:
			with open("/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/rchi2_progress.txt"%file_number, "a") as myfile:
				myfile.write('%d, %.3f time: %.2f \n'%(step, bdt_reduced_chi2, time_so_far))
		else:
			with open("test_output/rchi2_progress.txt", "a") as myfile:
				myfile.write('%d, %.3f time: %.2f \n'%(step, bdt_reduced_chi2, time_so_far))

		return bdt_rchi2_list, bdt_sum_overlap_list


if __name__ == '__main__':

	if blue_crystal == True:
		X_train = np.load('/mnt/storage/scratch/am13743/gan_training_data/sample_neg13_x_x_broad.npy')
	else:
		X_train = np.load('sample_neg13_x_x_broad.npy')

	# Split weight data off.
	muon_weights, X_train = np.split(X_train, [1], axis=1)

	muon_weights = np.squeeze(muon_weights/np.sum(muon_weights))

	# plt.hist(muon_weights, bins=50)
	# plt.savefig('weights.png')
	# plt.close('all')

	# save_X_train = X_train[:100000]
	# def save_samples(index):
	# 	list_1 = np.empty(0)
	# 	for x in range(0, np.shape(save_X_train)[0]):
	# 		list_1 = np.append(list_1, save_X_train[x][index])
	# 	plt.hist(list_1,range=[-1,1], bins=250)
	# 	plt.savefig('/mnt/storage/scratch/am13743/low_memory_gan_out/%d/test_output/sample_%d.png'%(file_number,index))
	# 	plt.close('all') 
	# 	return 
	# for k in range(0, 6):
	# 	save_samples(k)
	# del save_X_train
	
	gan = GAN()

	sample_size = int(np.shape(X_train)[0])

	gan.train(X_train, muon_weights)
















