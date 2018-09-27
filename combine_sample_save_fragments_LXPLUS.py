'''
Pre-processing script combining multiple fragments of full simulation data.
Also combines min and max files to get overall min-max.
This file is no longer used as pre-processing the full data sample must be computed on a cluster.
'''
import numpy as np
import glob
import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt


files_neg13_0_0 = glob.glob('sample_save/neg13_0_0/frag*')
files_neg13_x_x = glob.glob('sample_save/neg13_x_x/frag*')
files_pos13_0_0 = glob.glob('sample_save/pos13_0_0/frag*')
files_pos13_x_x = glob.glob('sample_save/pos13_x_x/frag*')

min_max_neg13_0_0 = glob.glob('sample_save/neg13_0_0/min_max*')
min_max_neg13_x_x = glob.glob('sample_save/neg13_x_x/min_max*')
min_max_pos13_0_0 = glob.glob('sample_save/pos13_0_0/min_max*')
min_max_pos13_x_x = glob.glob('sample_save/pos13_x_x/min_max*')

sample_neg13_0_0 = np.empty((0,5))
sample_neg13_x_x = np.empty((0,7))
sample_pos13_0_0 = np.empty((0,5))
sample_pos13_x_x = np.empty((0,7))

min_max_total_neg13_0_0 = np.empty((0,4,2))
min_max_total_neg13_x_x = np.empty((0,6,2))
min_max_total_pos13_0_0 = np.empty((0,4,2))
min_max_total_pos13_x_x = np.empty((0,6,2))

full_min_max_neg13_0_0 = np.empty((4,2))
full_min_max_neg13_x_x = np.empty((6,2))
full_min_max_pos13_0_0 = np.empty((4,2))
full_min_max_pos13_x_x = np.empty((6,2))

###############
print('Saving neg13 0 0')

for file in files_neg13_0_0:
	current = np.load(file)
	sample_neg13_0_0 = np.concatenate((current, sample_neg13_0_0),axis=0)

for file in min_max_neg13_0_0:
	current = np.load(file)
	min_max_total_neg13_0_0 = np.append(min_max_total_neg13_0_0, [current], axis=0)

for x in range(0, 4):
	full_min_max_neg13_0_0[x][0] = np.amin(min_max_total_neg13_0_0[:,x,0])
	full_min_max_neg13_0_0[x][1] = np.amax(min_max_total_neg13_0_0[:,x,1])

range_values = np.empty(4)

for x in range(0, 4):
	range_values[x] = full_min_max_neg13_0_0[x][1]-full_min_max_neg13_0_0[x][0]

for x in range(0, np.shape(sample_neg13_0_0)[0]):
	for index in range(0, 4):
		sample_neg13_0_0[x][index+1] = ((sample_neg13_0_0[x][index+1] - full_min_max_neg13_0_0[index][0])/range_values[index]) * 2 - 1

print('Saved',np.shape(sample_neg13_0_0)[0],'neg13 0 0 muons.')

np.save('sample_save/sample_neg13_0_0',sample_neg13_0_0)
np.save('sample_save/min_max_neg13_0_0',full_min_max_neg13_0_0)

del sample_neg13_0_0, full_min_max_neg13_0_0
################
# quit()

################
print('Saving pos13 0 0')

for file in files_pos13_0_0:
	current = np.load(file)
	sample_pos13_0_0 = np.concatenate((current, sample_pos13_0_0),axis=0)

for file in min_max_pos13_0_0:
	current = np.load(file)
	min_max_total_pos13_0_0 = np.append(min_max_total_pos13_0_0, [current], axis=0)

for x in range(0, 4):
	full_min_max_pos13_0_0[x][0] = np.amin(min_max_total_pos13_0_0[:,x,0])
	full_min_max_pos13_0_0[x][1] = np.amax(min_max_total_pos13_0_0[:,x,1])

range_values = np.empty(4)

for x in range(0, 4):
	range_values[x] = full_min_max_pos13_0_0[x][1]-full_min_max_pos13_0_0[x][0]

for x in range(0, np.shape(sample_pos13_0_0)[0]):
	for index in range(0, 4):
		sample_pos13_0_0[x][index+1] = ((sample_pos13_0_0[x][index+1] -full_min_max_pos13_0_0[index][0])/range_values[index]) * 2 - 1

print('Saved',np.shape(sample_pos13_0_0)[0],'pos13 0 0 muons.')

np.save('sample_save/sample_pos13_0_0',sample_pos13_0_0)
np.save('sample_save/min_max_pos13_0_0',full_min_max_pos13_0_0)

del sample_pos13_0_0, full_min_max_pos13_0_0
#################


################
print('Saving neg13 x x')

for file in files_neg13_x_x:
	current = np.load(file)
	sample_neg13_x_x = np.concatenate((current, sample_neg13_x_x),axis=0)

for file in min_max_neg13_x_x:
	current = np.load(file)
	min_max_total_neg13_x_x = np.append(min_max_total_neg13_x_x, [current], axis=0)

for x in range(0, 6):
	full_min_max_neg13_x_x[x][0] = np.amin(min_max_total_neg13_x_x[:,x,0])
	full_min_max_neg13_x_x[x][1] = np.amax(min_max_total_neg13_x_x[:,x,1])

range_values = np.empty(6)

for x in range(0, 6):
	range_values[x] = full_min_max_neg13_x_x[x][1]-full_min_max_neg13_x_x[x][0]

for x in range(0, np.shape(sample_neg13_x_x)[0]):
	for index in range(0, 6):
		sample_neg13_x_x[x][index+1] = ((sample_neg13_x_x[x][index+1] -full_min_max_neg13_x_x[index][0])/range_values[index]) * 2 - 1

print('Saved',np.shape(sample_neg13_x_x)[0],'neg13 x x muons.')

###############
broaden = True
save_dangerous = True


if broaden == True:
	min_max_2 = np.empty((2,2))
	range_values_2 = np.empty(2)
	means = np.empty(2)

	for y in range(0, 2):
		mean = np.mean(sample_neg13_x_x[:,y+1])
		
		means[y-1] = mean

		sample_neg13_x_x[:,y+1] = sample_neg13_x_x[:,y+1] - mean

		# print(np.mean(sample_neg13_x_x[:,y]))

		for x in range(0, np.shape(sample_neg13_x_x)[0]):
			if sample_neg13_x_x[x][y+1] < 0:
				sample_neg13_x_x[x][y+1] = -1 * (abs(sample_neg13_x_x[x][y+1])**0.5) 
			if sample_neg13_x_x[x][y+1] > 0:
				sample_neg13_x_x[x][y+1] = abs(sample_neg13_x_x[x][y+1])**0.5 
			sample_neg13_x_x[x][y+1] = sample_neg13_x_x[x][y+1] + mean


		min_max_2[y][0] = np.amin(sample_neg13_x_x[:,y+1])
		min_max_2[y][1] = np.amax(sample_neg13_x_x[:,y+1])
		range_values_2[y] = min_max_2[y][1] - min_max_2[y][0]

		for x in range(0, np.shape(sample_neg13_x_x)[0]):
			sample_neg13_x_x[x][y+1] = ((sample_neg13_x_x[x][y+1] - min_max_2[y][0])/range_values_2[y]) * 2 - 1

	np.save('sample_save/min_max_neg13_x_x',full_min_max_neg13_x_x)
	np.save('sample_save/min_max_2_neg13_x_x',min_max_2)
	np.save('sample_save/sample_neg13_x_x_broad',sample_neg13_x_x)
	np.save('sample_save/means_neg13_x_x', means)

else:
	np.save('sample_save/sample_neg13_x_x',sample_neg13_x_x)
	np.save('sample_save/min_max_neg13_x_x',full_min_max_neg13_x_x)
	
del sample_neg13_x_x, full_min_max_neg13_x_x
# #################


# ################
print('Saving pos13 x x')

for file in files_pos13_x_x:
	current = np.load(file)
	sample_pos13_x_x = np.concatenate((current, sample_pos13_x_x),axis=0)

for file in min_max_pos13_x_x:
	current = np.load(file)
	min_max_total_pos13_x_x = np.append(min_max_total_pos13_x_x, [current], axis=0)

for x in range(0, 6):
	full_min_max_pos13_x_x[x][0] = np.amin(min_max_total_pos13_x_x[:,x,0])
	full_min_max_pos13_x_x[x][1] = np.amax(min_max_total_pos13_x_x[:,x,1])

range_values = np.empty(6)

for x in range(0, 6):
	range_values[x] = full_min_max_pos13_x_x[x][1]-full_min_max_pos13_x_x[x][0]

for x in range(0, np.shape(sample_pos13_x_x)[0]):
	for index in range(0, 6):
		sample_pos13_x_x[x][index+1] = ((sample_pos13_x_x[x][index+1] -full_min_max_pos13_x_x[index][0])/range_values[index]) * 2 - 1

print('Saved',np.shape(sample_pos13_x_x)[0],'pos13 x x muons.')

broaden = True
save_dangerous = True


if broaden == True:
	min_max_2 = np.empty((2,2))
	range_values_2 = np.empty(2)
	means = np.empty(2)

	for y in range(0, 2):
		mean = np.mean(sample_pos13_x_x[:,y+1])
		
		means[y] = mean

		sample_pos13_x_x[:,y+1] = sample_pos13_x_x[:,y+1] - mean

		for x in range(0, np.shape(sample_pos13_x_x)[0]):
			if sample_pos13_x_x[x][y+1] < 0:
				sample_pos13_x_x[x][y+1] = -1 * (abs(sample_pos13_x_x[x][y+1])**0.5) 
			if sample_pos13_x_x[x][y+1] > 0:
				sample_pos13_x_x[x][y+1] = abs(sample_pos13_x_x[x][y+1])**0.5 
			sample_pos13_x_x[x][y+1] = sample_pos13_x_x[x][y+1] + mean


		min_max_2[y][0] = np.amin(sample_pos13_x_x[:,y+1])
		min_max_2[y][1] = np.amax(sample_pos13_x_x[:,y+1])
		range_values_2[y] = min_max_2[y][1] - min_max_2[y][0]

		for x in range(0, np.shape(sample_pos13_x_x)[0]):
			sample_pos13_x_x[x][y+1] = ((sample_pos13_x_x[x][y+1] - min_max_2[y][0])/range_values_2[y]) * 2 - 1

	np.save('sample_save/min_max_pos13_x_x',full_min_max_pos13_x_x)
	np.save('sample_save/min_max_2_pos13_x_x',min_max_2)
	np.save('sample_save/sample_pos13_x_x_broad',sample_pos13_x_x)
	np.save('sample_save/means_pos13_x_x', means)

else:
	np.save('sample_save/sample_pos13_x_x',sample_pos13_x_x)
	np.save('sample_save/min_max_pos13_x_x',full_min_max_pos13_x_x)
	
del sample_pos13_x_x, full_min_max_pos13_x_x
#################















