import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from matplotlib.colors import LogNorm
import math
import argparse
from keras.optimizers import Adam, RMSprop
import time
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from matplotlib.colors import LogNorm
from keras.models import Model, Sequential

file_number = 0
blue_crystal = False
include_batchnorm_and_dropout = True
# Hyperparameters

original_dim = 6
nb_epoch = 1
epochs= 10000000
epsilon_std = 1.0

parser = argparse.ArgumentParser()

parser.add_argument('-b', action='store', dest='batch_size', type=int,
                    help='batch size', default=100)

parser.add_argument('-s', action='store', dest='save_interval', type=int,
                    default = 10000,
                    help='save_interval')

parser.add_argument('-x', action='store', default=1000, type=float,
                    dest='xent_factor',
                    help='xent_factor')

parser.add_argument('-iE', action='store', default=100, type=int,
                    dest='intermediate_dimE')

parser.add_argument('-iD', action='store', default=100, type=int,
                    dest='intermediate_dimD')

parser.add_argument('-l', action='store', default=4, type=int,
                    dest='latent_dim',
                    help='latent_dim')

parser.add_argument('-o', action='store', default=0, type=int,
                    dest='optimizer_choice',
                    help='optimizer_choice')

parser.add_argument('-lr', action='store', default=0.001, type=float,
                    dest='learning_rate',
                    help='learning_rate')

parser.add_argument('-layersE', action='store', default=3, type=int,
                    dest='layersE')

parser.add_argument('-layersD', action='store', default=3, type=int,
                    dest='layersD')

results = parser.parse_args()

batch_size = batch = results.batch_size
save_interval = results.save_interval
xent_factor = results.xent_factor
intermediate_dimE = results.intermediate_dimE
intermediate_dimD = results.intermediate_dimD
layersD = results.layersD
layersE = results.layersE
latent_dim = results.latent_dim
optimizer_choice = results.optimizer_choice
learning_rate = results.learning_rate


loss = np.empty((0,2))

if include_batchnorm_and_dropout == False:
    if layersD == 1:
        decoder = Sequential([
            Dense(intermediate_dimD, input_dim=latent_dim, activation='relu'),
            Dense(original_dim, activation='sigmoid')
        ])
    elif layersD == 2:
        decoder = Sequential([
            Dense(intermediate_dimD, input_dim=latent_dim, activation='relu'),
            Dense(intermediate_dimD, activation='relu'),
            Dense(original_dim, activation='sigmoid')
        ])
    elif layersD == 3:
        decoder = Sequential([
            Dense(intermediate_dimD, input_dim=latent_dim, activation='relu'),
            Dense(intermediate_dimD, activation='relu'),
            Dense(intermediate_dimD, activation='relu'),
            Dense(original_dim, activation='sigmoid')
        ])

    if layersE == 1:
        x = Input(shape=(original_dim,))
        h = Dense(intermediate_dimE, activation='relu')(x)
    if layersE == 2:
        x = Input(shape=(original_dim,))
        h_2 = Dense(intermediate_dimE, activation='relu')(x)
        h = Dense(intermediate_dimE, activation='relu')(h_2)
    if layersE == 3:
        x = Input(shape=(original_dim,))
        h_2 = Dense(intermediate_dimE, activation='relu')(x)
        h_3 = Dense(intermediate_dimE, activation='relu')(h_2)
        h = Dense(intermediate_dimE, activation='relu')(h_3)
else:
    if layersD == 1:
        decoder = Sequential([
            Dense(intermediate_dimD, input_dim=latent_dim, activation='relu'),
            Dropout(0.25),
            BatchNormalization(momentum=0.8),
            Dense(original_dim, activation='sigmoid')
        ])
    elif layersD == 2:
        decoder = Sequential([
            Dense(intermediate_dimD, input_dim=latent_dim, activation='relu'),
            Dropout(0.25),
            BatchNormalization(momentum=0.8),
            Dense(intermediate_dimD, activation='relu'),
            Dropout(0.25),
            BatchNormalization(momentum=0.8),
            Dense(original_dim, activation='sigmoid')
        ])
    elif layersD == 3:
        decoder = Sequential([
            Dense(intermediate_dimD, input_dim=latent_dim, activation='relu'),
            Dropout(0.25),
            BatchNormalization(momentum=0.8),
            Dense(intermediate_dimD, activation='relu'),
            Dropout(0.25),
            BatchNormalization(momentum=0.8),
            Dense(intermediate_dimD, activation='relu'),
            Dropout(0.25),
            BatchNormalization(momentum=0.8),
            Dense(original_dim, activation='sigmoid')
        ])

    if layersE == 1:
        x = Input(shape=(original_dim,))

        x_2 = Dropout(0.25)(x)
        x_3 = BatchNormalization(momentum=0.8)(x_2)

        h = Dense(intermediate_dimE, activation='relu')(x_3)
    if layersE == 2:
        x = Input(shape=(original_dim,))

        x_2 = Dropout(0.25)(x)
        x_3 = BatchNormalization(momentum=0.8)(x_2)

        h_2 = Dense(intermediate_dimE, activation='relu')(x_3)

        x_4 = Dropout(0.25)(h_2)
        h_3 = BatchNormalization(momentum=0.8)(x_4)

        h = Dense(intermediate_dimE, activation='relu')(h_3)
    if layersE == 3:
        x = Input(shape=(original_dim,))

        x_2 = Dropout(0.25)(x)
        x_3 = BatchNormalization(momentum=0.8)(x_2)

        h_2 = Dense(intermediate_dimE, activation='relu')(x_3)

        x_6 = Dropout(0.25)(h_2)
        x_7 = BatchNormalization(momentum=0.8)(x_6)

        h_3 = Dense(intermediate_dimE, activation='relu')(x_7)

        x_8 = Dropout(0.25)(h_3)
        x_9 = BatchNormalization(momentum=0.8)(x_8)

        h = Dense(intermediate_dimE, activation='relu')(x_9)



z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0, stddev=1)
    print(epsilon)
    return z_mean + K.exp(z_log_var / 2) * epsilon
    
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

x_decoded_mean = decoder(z)

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss*xent_factor + kl_loss

vae = Model(x, x_decoded_mean)

optimizer = Adam(lr=learning_rate, beta_1=0.5, decay=8e-8, amsgrad=False)
vae.compile(optimizer=optimizer, loss=vae_loss)

vae.summary()

if blue_crystal == True:
    X_train = np.load('/mnt/storage/scratch/am13743/gan_training_data/sample_neg13_x_x_broad.npy')
else:
    X_train = np.load('/Users/am13743/Desktop/GANs/thomas sample/sample_neg13_x_x_broad.npy')

muon_weights, X_train = np.split(X_train, [1], axis=1)

muon_weights = np.squeeze(muon_weights)

for xx in range(0, np.shape(X_train)[0]):
    for y in range(0, 6):
        X_train[xx][y] = (X_train[xx][y] + 1)/2

x_test = X_train[:10000]
y_test = x_test
X_train = X_train[:1880000] 
muon_weights = muon_weights[:1880000]
muon_weights = muon_weights/np.sum(muon_weights)

list_for_np_choice = np.arange(np.shape(X_train)[0]) 
y_train = X_train

bdt_sum_overlap_list = np.empty((0,2))
bdt_rchi2_list = np.empty((0,2))

t0 = time.time()

for epoch in range(0, epochs):

    # random_index = np.random.randint(0, len(X_train) - batch)
    # legit_images = X_train[random_index : random_index + int(batch)].reshape(int(batch), 6)

    random_indicies = np.random.choice(list_for_np_choice, size=batch, p=muon_weights, replace=False)
    legit_images = X_train[random_indicies].reshape(int(batch), 6)

    vae_loss_value = vae.train_on_batch(legit_images, legit_images)

    if epoch % 100 == 0 :print('Epoch:',epoch,', VAE Loss:',vae_loss_value)

    loss = np.append(loss, [[epoch, vae_loss_value]], axis=0)



    save = False

    if epoch % save_interval == 0 and epoch > 1: save = True

    if save == True:

        # print(np.shape(X_train))
        # latent_space = encoder.predict(X_train[:100])

        # plt.hist([latent_space[:,0],latent_space[:,1],latent_space[:,2],latent_space[:,3]], bins=50)
        # plt.savefig('test_output/latent_space.png')
        # plt.close('all')


        plt.figure(figsize=(6, 6))
        plt.plot(loss[:,0], loss[:,1])
        if blue_crystal == True:
            plt.savefig('/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/Loss.png'%file_number)
        else:
            plt.savefig('test_output/Loss.png')
        plt.close('all')

        sample_test_size = np.shape(X_train)[0]
        sampling_test = np.empty((sample_test_size, latent_dim))

        for num in range(0, sample_test_size):
            for index in range(0, latent_dim):
                sampling_test[num][index] = np.random.normal(0, 1)

        images = decoder.predict(sampling_test)

        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

        bdt_train_size = 50000

        real_training_data = X_train[:bdt_train_size]

        real_test_data = X_train[bdt_train_size:bdt_train_size*2]

        fake_training_data = np.squeeze(images[:bdt_train_size])

        fake_test_data = np.squeeze(images[bdt_train_size:bdt_train_size*2])

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
            plt.savefig('/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/bdt/BDT_out_%d.png'%(file_number,epoch), bbox_inches='tight')
        else:
            plt.savefig('test_output/bdt/BDT_out_%d.png'%(epoch), bbox_inches='tight')
        plt.close('all')


        real_hist = np.histogram(out_real[:,1],bins=100,range=[0,1])
        fake_hist = np.histogram(out_fake[:,1],bins=100,range=[0,1])

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

        bdt_rchi2_list = np.append(bdt_rchi2_list, [[epoch, bdt_reduced_chi2]], axis=0)

        plt.plot(bdt_rchi2_list[:,0], bdt_rchi2_list[:,1])
        plt.yscale('log', nonposy='clip')
        if blue_crystal == True:
            plt.savefig('/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/bdt_rchi2.png'%file_number, bbox_inches='tight')
        else:
            plt.savefig('/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/bdt_rchi2.png'%file_number, bbox_inches='tight')
        plt.close('all')

        sum_overlap = 0

        for bin_index in range(0, 100):
            if real_hist[0][bin_index] < fake_hist[0][bin_index]:
                sum_overlap += real_hist[0][bin_index]
            elif fake_hist[0][bin_index] < real_hist[0][bin_index]:
                sum_overlap += fake_hist[0][bin_index]
            else:
                sum_overlap += fake_hist[0][bin_index]

        bdt_sum_overlap_list = np.append(bdt_sum_overlap_list, [[epoch, sum_overlap]], axis=0)

        plt.plot(bdt_sum_overlap_list[:,0], bdt_sum_overlap_list[:,1])
        plt.yscale('log', nonposy='clip')
        if blue_crystal == True:
            plt.savefig('/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/bdt_overlap.png'%file_number, bbox_inches='tight')
        else:
            plt.savefig('/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/bdt_overlap.png'%file_number, bbox_inches='tight')
        plt.close('all')

        t1 = time.time()
        time_so_far = t1-t0

        if blue_crystal == True:
            with open("/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/convergence_details.txt"%file_number, "a") as myfile:
                myfile.write('\n epoch: %d, time: %.3f, rchi2: %.3f, overlap: %.3f'%(epoch, time_so_far, bdt_reduced_chi2, sum_overlap))
        else:
            with open("test_output/convergence_details.txt", "a") as myfile:
                myfile.write('\n epoch: %d, time: %.3f, rchi2: %.3f, overlap: %.3f'%(epoch, time_so_far, bdt_reduced_chi2, sum_overlap))
        
        def compare_cross_hists_2d_hist(index_1, index_2):
            plt.subplot(1,2,1)
            plt.title('Real Properties')
            plt.hist2d(X_train[:100000,index_1],X_train[:100000,index_2], bins = 100, norm=LogNorm(), range=[[0,1],[0,1]])
            plt.subplot(1,2,2)
            plt.title('Generated blur')
            plt.hist2d(images[:100000,index_1],images[:100000,index_2], bins = 100, norm=LogNorm(), range=[[0,1],[0,1]])
            if blue_crystal == True:
                plt.savefig('/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/compare_cross_blur_%d_%d_2D.png'%(file_number,index_1, index_2), bbox_inches='tight')
            else:
                plt.savefig('test_output/compare_cross_blur_%d_%d_2D.png'%(index_1, index_2), bbox_inches='tight')
            plt.close('all')

        for x in range(0,6):
            for y in range(x+1,6):
                compare_cross_hists_2d_hist(x,y)

        for index in range(0, 6):
            plt.hist([X_train[:100000,index],images[:100000,index]], bins = 250,label=['real','gen'], histtype='step')
            plt.legend(loc='upper right')
            if blue_crystal == True:
                plt.savefig('/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/%d/out.png'%(file_number,index), bbox_inches='tight')
            else:
                plt.savefig('test_output/%d/out.png'%(index), bbox_inches='tight')
            plt.close('all')

        for index in range(0, 6):
            plt.hist([X_train[:100000,index],images[:100000,index]], bins = 250,label=['real','gen'], histtype='step')
            plt.legend(loc='upper right')
            if blue_crystal == True:
                plt.savefig('/mnt/storage/scratch/am13743/low_memory_vae_out_6/%d/test_output/current_%d.png'%(file_number,index), bbox_inches='tight')
            else:
                plt.savefig('test_output/current_%d.png'%(index), bbox_inches='tight')
            plt.close('all')






