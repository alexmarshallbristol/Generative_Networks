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


# G = load_model('Generator_neg13_x_x_best_overlap.h5', custom_objects = {'_loss_generator':_loss_generator})
G = load_model('Generator_pos13_x_x_current.h5', custom_objects = {'_loss_generator':_loss_generator})

G.summary()

X_train = np.load('/Users/am13743/Desktop/GANs/thomas sample/09/sample_pos13_x_x_broad.npy')

means = np.load('/Users/am13743/Desktop/GANs/thomas sample/09/means_pos13_x_x.npy')
min_max_1 = np.load('/Users/am13743/Desktop/GANs/thomas sample/09/min_max_pos13_x_x.npy')
min_max_2 = np.load('/Users/am13743/Desktop/GANs/thomas sample/09/min_max_2_pos13_x_x.npy')

# print(min_max_2)
# quit()




muon_weights, X_train = np.split(X_train, [1], axis=1)
muon_weights = np.squeeze(muon_weights/np.sum(muon_weights))
list_for_np_choice = np.arange(np.shape(X_train)[0]) 


# noise_size = 100000

bdt_train_size = 50000

noise = np.random.normal(0, 1, (bdt_train_size, 100))

images = G.predict(noise)

# Initialise BDT classifier used to check progress.
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

# Create training samples for BDT.
random_indicies = np.random.choice(list_for_np_choice, size=(2,50000), p=muon_weights, replace=False)

real_training_data = X_train[random_indicies[0]]

# real_test_data = X_train[random_indicies[1]]

fake_training_data = np.squeeze(images[:bdt_train_size])

# fake_test_data = np.squeeze(images[bdt_train_size:])

real_training_labels = np.ones(bdt_train_size)

fake_training_labels = np.zeros(bdt_train_size)

total_training_data = np.concatenate((real_training_data, fake_training_data))

total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

clf.fit(total_training_data, total_training_labels)



noise_2 = np.random.normal(0, 1, (100000, 100))

images_2 = G.predict(noise_2)

images_2 = np.squeeze(images_2)

#plot prob against p



x_prob_hist = np.empty((0,2))



def normalize_un_broaden(input_array,index):
	print(np.shape(input_array))

	print(min_max_2[index][1],min_max_2[index][0])

	print(min_max_1[index][1],min_max_1[index][0])

	for x in range(0, np.shape(input_array)[0]):
		input_array[x] = (((input_array[x]+0.97)/1.94)*(min_max_2[index][1] - min_max_2[index][0])+ min_max_2[index][0])
		input_array[x] = input_array[x] - means[index]
		if input_array[x] < 0:
			input_array[x] = -(input_array[x]**2)
		if input_array[x] > 0:
			input_array[x] = (input_array[x]**2)
		input_array[x] = input_array[x] + means[index]
		input_array[x] = (((input_array[x]+1)/2)*(min_max_1[index][1] - min_max_1[index][0])+ min_max_1[index][0])



	return input_array

def normalize(input_array,index):


	for x in range(0, np.shape(input_array)[0]):

		input_array[x] = (((input_array[x]+1)/2)*(min_max_1[index][1] - min_max_1[index][0])+ min_max_1[index][0])



	return input_array

prob = clf.predict_proba(images_2[:])

out_x = normalize_un_broaden(images_2[:,0],0)
plt.hist2d(out_x, prob[:,1] ,bins=50, norm=LogNorm(), cmap='jet')
plt.savefig('test_x.png')
plt.close('all')

out_y = normalize_un_broaden(images_2[:,1],1)
plt.hist2d(out_y, prob[:,1] ,bins=50, norm=LogNorm(), cmap='jet')
plt.savefig('test_y.png')
plt.close('all')

out_z = normalize(images_2[:,2],2)
plt.hist2d(out_z, prob[:,1] ,bins=50, norm=LogNorm(), cmap='jet')
plt.savefig('test_z.png')
plt.close('all')

out_px = normalize(images_2[:,3],3)
plt.hist2d(out_px, prob[:,1] ,bins=50, norm=LogNorm(), cmap='jet')
plt.savefig('test_px.png')
plt.close('all')

out_py = normalize(images_2[:,4],4)
plt.hist2d(out_py, prob[:,1] ,bins=50, norm=LogNorm(), cmap='jet')
plt.savefig('test_py.png')
plt.close('all')

out_pz = normalize(images_2[:,5],5)
plt.hist2d(out_pz, prob[:,1] ,bins=50, norm=LogNorm(), cmap='jet')
plt.savefig('test_pz.png')
plt.close('all')



quit()











