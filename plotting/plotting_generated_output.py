'''
Plotting file for use after combined_generated_npy.py
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


real_data = np.load('/Users/am13743/Desktop/GANs/fudge/real_data.npy')

gen_raw = np.load('/Users/am13743/Desktop/GANs/fudge/gen_raw2_full.npy')

gen_raw = gen_raw[:np.shape(real_data)[0]]

plt.figure(figsize=(10,7))
plt.subplot(2,3,1)
plt.hist([real_data[:,0], gen_raw[:,0]],normed=True, bins=100, color=['#4772FF','#FF5959'], linewidth='1.5',histtype='step')
plt.xlabel('$x$ (cm)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
plt.yscale('log')

plt.subplot(2,3,2)
plt.hist([real_data[:,1], gen_raw[:,1]],normed=True, bins=100, color=['#4772FF','#FF5959'], linewidth='1.5',histtype='step')
plt.xlabel('$y$ (cm)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
plt.yscale('log')

plt.subplot(2,3,3)
plt.hist([real_data[:,2], gen_raw[:,2]],normed=True, bins=100, color=['#4772FF','#FF5959'], linewidth='1.5',label=('Full Simulation','Generated'),histtype='step')
plt.xlabel('$z$ (cm)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
# plt.yscale('log')

plt.legend(loc='upper right')
plt.subplot(2,3,4)
plt.hist([real_data[:,3], gen_raw[:,3]],normed=True, bins=100, color=['#4772FF','#FF5959'], linewidth='1.5',histtype='step')
plt.xlabel('$P_x$ (GeV)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
# plt.yscale('log')

plt.subplot(2,3,5)
plt.hist([real_data[:,4], gen_raw[:,4]],normed=True, bins=100, color=['#4772FF','#FF5959'], linewidth='1.5',histtype='step')
plt.xlabel('$P_y$ (GeV)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
# plt.yscale('log')

plt.subplot(2,3,6)
plt.hist([real_data[:,5], gen_raw[:,5]],normed=True, bins=100, color=['#4772FF','#FF5959'], linewidth='1.5',histtype='step')
plt.xlabel('$P_z$ (GeV)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
# plt.yscale('log')

plt.savefig('plots/1D.png')
plt.close('all')


gen_z_hist = np.histogram(gen_raw[:,2], bins=50, range=[-7150, -6850])
plot_gen_z_hist = np.empty((0,3))
for i in range(0, np.shape(gen_z_hist[0])[0]):
	plot_gen_z_hist = np.append(plot_gen_z_hist, [[(gen_z_hist[1][i]+gen_z_hist[1][i+1])/2,gen_z_hist[0][i],math.sqrt(gen_z_hist[0][i])]], axis=0)


real_z_hist = np.histogram(real_data[:,2], bins=50, range=[-7150, -6850])
plot_real_z_hist = np.empty((0,3))
for i in range(0, np.shape(real_z_hist[0])[0]):
	plot_real_z_hist = np.append(plot_real_z_hist, [[(real_z_hist[1][i]+real_z_hist[1][i+1])/2,real_z_hist[0][i],math.sqrt(real_z_hist[0][i])]], axis=0)

# print(np.shape(plot_gen_mom_hist), np.shape(plot_real_mom_hist), np.shape(plot_gen_tmom_hist), np.shape(plot_gen_tmom_hist))

plt.errorbar(plot_gen_z_hist[:,0], plot_gen_z_hist[:,1], yerr=plot_gen_z_hist[:,2], color='#FF5959',label='Generated', capsize=4, marker='s',markersize=3,linewidth=0,elinewidth=2)
plt.errorbar(plot_real_z_hist[:,0], plot_real_z_hist[:,1], yerr=plot_real_z_hist[:,2], color='#4772FF',label='Full Simulation', capsize=4, marker='s',markersize=3,linewidth=0,elinewidth=2)
# plt.yscale('log')
plt.legend(loc='upper right')
plt.xlabel('Start of Track in Z (cm)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
plt.tick_params(axis='x', which='both', labelsize=7)
plt.savefig('plots/Z.png')
plt.close('all')






#martix maths multiplication here to get p and pt
pos, mom_real = np.split(real_data, [3], axis=1)
p_real = mom_real.sum(axis=1)
p_x_real, p_y_real, p_z_real = np.split(mom_real, [1,2], axis=1)
p_x_sq_real = np.multiply(p_x_real, p_x_real)
p_y_sq_real = np.multiply(p_y_real, p_y_real)
sqs_array_real = np.concatenate((p_x_sq_real,p_y_sq_real),axis=1)
sum_sqs_real = sqs_array_real.sum(axis=1)
p_t_real = np.sqrt(sum_sqs_real)
p_real = np.expand_dims(p_real,1)
p_t_real = np.expand_dims(p_t_real,1)
plot_real_data = np.concatenate((p_real, p_t_real),axis=1)

pos, mom_raw = np.split(gen_raw, [3], axis=1)
p_raw = mom_raw.sum(axis=1)
p_x_raw, p_y_raw, p_z_raw = np.split(mom_raw, [1,2], axis=1)
p_x_sq_raw = np.multiply(p_x_raw, p_x_raw)
p_y_sq_raw = np.multiply(p_y_raw, p_y_raw)
sqs_array_raw = np.concatenate((p_x_sq_raw,p_y_sq_raw),axis=1)
sum_sqs_raw = sqs_array_raw.sum(axis=1)
p_t_raw = np.sqrt(sum_sqs_raw)
p_raw = np.expand_dims(p_raw,1)
p_t_raw = np.expand_dims(p_t_raw,1)
plot_gen_raw = np.concatenate((p_raw, p_t_raw),axis=1)


plt.figure(figsize=(7.5,3))
plt.subplot(1,2,2)
plt.title('Generated')
# plt.hist2d(plot_gen_raw[:,0], plot_gen_raw[:,1], bins=100, norm=LogNorm(), range=[[np.amin(plot_gen_raw[:,0]), np.amax(plot_gen_raw[:,0])],[np.amin(plot_gen_raw[:,1]), np.amax(plot_gen_raw[:,1])]], cmap=plt.cm.CMRmap)
plt.hist2d(plot_gen_raw[:,0], plot_gen_raw[:,1], bins=100, norm=LogNorm(), range=[[0,400],[0,6]], cmap=plt.cm.CMRmap)
# plt.ylabel('Transverse Momentum (GeV)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
plt.tick_params(axis='x', which='both', labelsize=7)
plt.xlabel('Momentum (GeV)', fontsize='small')
plt.subplot(1,2,1)
plt.title('Full Simulation')
# plt.hist2d(plot_real_data[:,0], plot_real_data[:,1], bins=100, norm=LogNorm(), range=[[np.amin(plot_gen_raw[:,0]), np.amax(plot_gen_raw[:,0])],[np.amin(plot_gen_raw[:,1]), np.amax(plot_gen_raw[:,1])]], cmap=plt.cm.CMRmap)
plt.hist2d(plot_real_data[:,0], plot_real_data[:,1], bins=100, norm=LogNorm(), range=[[0,400],[0,6]], cmap=plt.cm.CMRmap)
plt.ylabel('Transverse Momentum (GeV)')
plt.xlabel('Momentum (GeV)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
plt.tick_params(axis='x', which='both', labelsize=7)

plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9, wspace = 0.1,hspace = 0.2)
cax = plt.axes([0.925, 0.1, 0.01, 0.8])
plt.colorbar(cax=cax)

plt.savefig('plots/test_mom.png',bbox_inches='tight')
plt.close('all')






#Z against P total

plt.figure(figsize=(7.5,3))
plt.subplot(1,2,2)
plt.title('Generated')
plt.hist2d(gen_raw[:,2], plot_gen_raw[:,0], bins=100, norm=LogNorm(), range=[[np.min(gen_raw[:,2]),np.max(real_data[:,2])],[0,np.max(plot_gen_raw[:,0])]],cmap=plt.cm.CMRmap)
# plt.hist2d(plot_gen_raw[:,0], gen_raw[:,2], bins=100, norm=LogNorm(), range=[[np.min(plot_gen_raw[:,0]),np.max(plot_gen_raw[:,0])],[np.min(gen_raw[:,2]),np.max(gen_raw[:,2])]],cmap=plt.cm.CMRmap)
plt.tick_params(axis='y', which='both', labelsize=7)
plt.tick_params(axis='x', which='both', labelsize=7)
plt.xlabel('Z (cm)', fontsize='small')
plt.subplot(1,2,1)
plt.title('Full Simulation')
plt.hist2d(real_data[:,2], plot_real_data[:,0], bins=100, norm=LogNorm(), range=[[np.min(gen_raw[:,2]),np.max(real_data[:,2])],[0,np.max(plot_gen_raw[:,0])]], cmap=plt.cm.CMRmap)
# plt.hist2d(plot_real_data[:,0], real_data[:,2], bins=100, norm=LogNorm(), range=[[np.min(plot_gen_raw[:,0]),np.max(plot_gen_raw[:,0])],[np.min(gen_raw[:,2]),np.max(gen_raw[:,2])]], cmap=plt.cm.CMRmap)
plt.ylabel('Momentum (GeV)')
plt.xlabel('Z (cm)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
plt.tick_params(axis='x', which='both', labelsize=7)

plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9, wspace = 0.2,hspace = 0.2)
cax = plt.axes([0.925, 0.1, 0.01, 0.8])
plt.colorbar(cax=cax)

plt.savefig('plots/total_mom_z.png',bbox_inches='tight')
plt.close('all')
# quit()


# print(phi)
# quit()



plt.figure(figsize=(0.5625*7.5,0.5625*3))
plt.subplot(1,2,2)
plt.title('Generated')
plt.hist2d(gen_raw[:,0], gen_raw[:,1], bins=100, norm=LogNorm(), range=[[-7,7],[-7,7]], cmap=plt.cm.CMRmap)
# plt.ylabel('$y$ (cm)', fontsize='small')
plt.xlabel('$x$ (cm)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=5)
plt.tick_params(axis='x', which='both', labelsize=5)
plt.subplot(1,2,1)
plt.title('Full Simulation')
plt.hist2d(real_data[:,0], real_data[:,1], bins=100, norm=LogNorm(), range=[[-7,7],[-7,7]], cmap=plt.cm.CMRmap)
plt.ylabel('$y$ (cm)')
plt.xlabel('$x$ (cm)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=5)
plt.tick_params(axis='x', which='both', labelsize=5)

# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9, wspace = 0.17,hspace = 0.2)
cax = plt.axes([0.925, 0.1, 0.01, 0.8])
plt.colorbar(cax=cax)

plt.savefig('plots/test2d_xy.png',bbox_inches='tight')
plt.close('all')




r = np.ones(np.shape(gen_raw)[0])*5+0.8*np.random.normal(size = np.shape(gen_raw)[0])
phi = np.random.uniform(size = np.shape(gen_raw)[0])*2*math.pi

cosphi = np.cos(phi)
sinphi = np.sin(phi)

dx = r * cosphi
dy = r * sinphi

gen_raw[:,0] = np.add(gen_raw[:,0], dx)
gen_raw[:,1] = np.add(gen_raw[:,1], dy)

real_data[:,0] = np.add(real_data[:,0], dx)
real_data[:,1] = np.add(real_data[:,1], dy)

plt.figure(figsize=(0.5625*7.5,0.5625*3))
plt.subplot(1,2,2)
plt.title('Generated')
plt.hist2d(gen_raw[:,0], gen_raw[:,1], bins=100, norm=LogNorm(), range=[[-14,14],[-14,14]], cmap=plt.cm.CMRmap)
# plt.ylabel('$y$ (cm)', fontsize='small')
plt.xlabel('$x$ (cm)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=5)
plt.tick_params(axis='x', which='both', labelsize=5)
plt.subplot(1,2,1)
plt.title('Full Simulation')
plt.hist2d(real_data[:,0], real_data[:,1], bins=100, norm=LogNorm(), range=[[-14,14],[-14,14]], cmap=plt.cm.CMRmap)
plt.ylabel('$y$ (cm)')
plt.xlabel('$x$ (cm)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=5)
plt.tick_params(axis='x', which='both', labelsize=5)

# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9, wspace = 0.17,hspace = 0.2)
cax = plt.axes([0.925, 0.1, 0.01, 0.8])
plt.colorbar(cax=cax)

plt.savefig('plots/test2d_xy_with_smearing.png',bbox_inches='tight')
plt.close('all')


# print(np.shape(plot_gen_raw[:,0]))
# print(np.where(plot_gen_raw[:,0] > 20), np.shape(np.where(plot_gen_raw[:,0] > 20)))

# plot_gen_raw = np.delete(plot_gen_raw, np.squeeze(np.where(plot_gen_raw[:,0] > 20)),axis=0)
# plot_real_data = np.delete(plot_real_data, np.squeeze(np.where(plot_real_data[:,0] > 20)),axis=0)

weights_raw = np.ones(np.shape(plot_gen_raw)[0])
weights_real = np.ones(np.shape(plot_real_data)[0])*(np.shape(plot_gen_raw[:,0])[0]/np.shape(plot_real_data[:,0])[0])
# weights_real = np.ones(np.shape(plot_real_data)[0]) 


# print(np.shape(plot_gen_raw[:,1]),np.shape(plot_gen_raw[:,0]),np.shape(plot_real_data[:,0]))

# print(weights_real)


gen_mom_hist = np.histogram(plot_gen_raw[:,0], bins=50, weights=weights_raw,range=[0, 360])
plot_gen_mom_hist = np.empty((0,3))
for i in range(0, np.shape(gen_mom_hist[0])[0]):
	plot_gen_mom_hist = np.append(plot_gen_mom_hist, [[(gen_mom_hist[1][i]+gen_mom_hist[1][i+1])/2,gen_mom_hist[0][i],math.sqrt(gen_mom_hist[0][i])]], axis=0)

real_mom_hist = np.histogram(plot_real_data[:,0], bins=50, weights=weights_real, range=[0, 360])
plot_real_mom_hist = np.empty((0,3))
for i in range(0, np.shape(real_mom_hist[0])[0]):
	plot_real_mom_hist = np.append(plot_real_mom_hist, [[(real_mom_hist[1][i]+real_mom_hist[1][i+1])/2,real_mom_hist[0][i],math.sqrt(real_mom_hist[0][i])]], axis=0)

gen_tmom_hist = np.histogram(plot_gen_raw[:,1], bins=50, weights=weights_raw, range=[0, 5])
plot_gen_tmom_hist = np.empty((0,3))
for i in range(0, np.shape(gen_tmom_hist[0])[0]):
	plot_gen_tmom_hist = np.append(plot_gen_tmom_hist, [[(gen_tmom_hist[1][i]+gen_tmom_hist[1][i+1])/2,gen_tmom_hist[0][i],math.sqrt(gen_tmom_hist[0][i])]], axis=0)


real_tmom_hist = np.histogram(plot_real_data[:,1], bins=50, weights=weights_real, range=[0, 5])
plot_real_tmom_hist = np.empty((0,3))
for i in range(0, np.shape(real_tmom_hist[0])[0]):
	plot_real_tmom_hist = np.append(plot_real_tmom_hist, [[(real_tmom_hist[1][i]+real_tmom_hist[1][i+1])/2,real_tmom_hist[0][i],math.sqrt(real_tmom_hist[0][i])]], axis=0)

# print(np.shape(plot_gen_mom_hist), np.shape(plot_real_mom_hist), np.shape(plot_gen_tmom_hist), np.shape(plot_gen_tmom_hist))



ratio_total_mom = plot_real_mom_hist[:,1]/plot_gen_mom_hist[:,1]
ratio_trans_mom = plot_real_tmom_hist[:,1]/plot_gen_tmom_hist[:,1]

# print(np.shape(ratio_total_mom))







plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.errorbar(plot_gen_mom_hist[:,0], plot_gen_mom_hist[:,1], yerr=plot_gen_mom_hist[:,2], color='#FF5959', capsize=4, marker='s',markersize=3,linewidth=0,elinewidth=2)
plt.errorbar(plot_real_mom_hist[:,0], plot_real_mom_hist[:,1], yerr=plot_real_mom_hist[:,2], color='#4772FF', capsize=4, marker='s',markersize=3,linewidth=0,elinewidth=2)
# plt.errorbar(plot_real_mom_hist[:,0], ratio_total_mom, color='#4772FF', capsize=4, marker='s',markersize=3,linewidth=0,elinewidth=2)
plt.yscale('log')
plt.xlabel('Momentum (GeV)', fontsize='small')
# plt.ylabel('Ratio REAL/GEN', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
plt.tick_params(axis='x', which='both', labelsize=7)

plt.subplot(1,2,2)
plt.errorbar(plot_gen_tmom_hist[:,0], plot_gen_tmom_hist[:,1], yerr=plot_gen_tmom_hist[:,2], color='#FF5959', capsize=4,label='Generated', marker='s',markersize=3,linewidth=0,elinewidth=2)
plt.errorbar(plot_real_tmom_hist[:,0], plot_real_tmom_hist[:,1], yerr=plot_real_tmom_hist[:,2], color='#4772FF', capsize=4,label='Full Simulation', marker='s',markersize=3,linewidth=0,elinewidth=2)
# plt.errorbar(plot_real_tmom_hist[:,0], ratio_trans_mom, color='#4772FF', capsize=4,label='Full Simulation', marker='s',markersize=3,linewidth=0,elinewidth=2)
plt.yscale('log')
plt.legend(loc='upper right')
plt.xlabel('Transverse Momentum (GeV)', fontsize='small')
plt.tick_params(axis='y', which='both', labelsize=7)
plt.tick_params(axis='x', which='both', labelsize=7)
# plt.savefig('plots/momentum_ratio.png')
plt.savefig('plots/momentum_er.png')
plt.close('all')




quit()




