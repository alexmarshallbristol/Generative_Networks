'''
Combined generated muon npy files into a single ready for plotting.
'''
import numpy as np

import math

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import glob


files = glob.glob('/Users/am13743/Desktop/GANs/final_all_muons/bc_out/full_muons*')

generated_training = np.empty((0,7))

for file in files:
	current = np.load(file)
	generated_training = np.append(generated_training, current ,axis=0)
	print(np.shape(generated_training))


charge, generated_training = np.split(generated_training, [1], axis=1)
print(np.shape(generated_training))

pos_array, mom_array= np.split(generated_training, [3], axis=1)


p = mom_array.sum(axis=1)

# take_off = (1/900)*p+0.05
# take_off = np.expand_dims(take_off,1)

p_x, p_y, p_z = np.split(mom_array, [1,2], axis=1)

p_x_sq = np.multiply(p_x, p_x)
p_y_sq = np.multiply(p_y, p_y)

sqs_array = np.concatenate((p_x_sq,p_y_sq),axis=1)

sum_sqs = sqs_array.sum(axis=1)

p_t = np.sqrt(sum_sqs)

p_t = np.expand_dims(p_t,1)

# factor = take_off/p_t

p_x, p_y, p_z = np.split(mom_array, [1,2], axis=1)

print(np.shape(pos_array), np.shape(p_x), np.shape(p_y), np.shape(p_z))
full_sample = np.concatenate((pos_array,p_x,p_y,p_z),axis=1)

np.save('gen_raw2_full', full_sample)






