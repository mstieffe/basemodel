import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d,lib.cnmem=0"%(random.randint(0,3))
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D, Deconvolution2D, Convolution3D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
import cPickle, random, sys, keras
from keras.models import Model
#from IPython import display
from keras.utils import np_utils
import keras.backend as K
import math
import tensorflow as tf

from config import *



opt = Adagrad(lr=0.001, epsilon=1e-08)



K.set_image_dim_ordering('th')


temp_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.04,1.1,1.2,1.3,1.4,1.5, 1.6]    

def make_noise_sample(scale, size):
    noise_sample = scale * np.random.randn(size, noise_dim)
    return noise_sample

def corr_out_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],100])

def meanplus_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],1])

def dist_corr(steps):
    dist = []
    for i in range(0,steps):
        for j in range(0,steps):
            dist.append(math.sqrt(i*i+j*j))
    return dist			
	
def correlation(layers):
    (N,dim,N_x,N_y) = layers.get_shape()
    steps = 10
    for i in range(0,steps):
        part1 = layers[:,:,:,0:int(N_y)-i]	
        part2 = layers[:,:,:,int(N_y)-i:]
        shifted_layer = tf.concat([part2,part1], 3)
        corr_temp = tf.multiply(layers, shifted_layer)
        corr_temp = tf.reduce_sum(corr_temp, 1, keep_dims = True)
        corr_temp = tf.reduce_sum(corr_temp, [2,3])
        corr_temp = tf.div(corr_temp, int(N_x)*int(N_y))
        for j in range(0,steps):
            part1 = shifted_layer[:,:,0:int(N_x)-j,:]
            part2 = shifted_layer[:,:,int(N_x)-j:,:]
            shifted_layer2 = tf.concat([part2,part1], 2)
            corr_temp = tf.multiply(layers, shifted_layer2)
            corr_temp = tf.reduce_sum(corr_temp, 1, keep_dims = True)
            corr_temp = tf.reduce_sum(corr_temp, [2,3])
            corr_temp = tf.div(corr_temp, int(N_x)*int(N_y))
            if (i+j) != 0:
                corr = tf.concat([corr, corr_temp], 1)	
            else:
                corr = corr_temp
    print(corr.get_shape())
    return corr
    
  
def pearson_corr(layers):
    (NN,N,dim,N_x,N_y) = layers.get_shape()
    l1 = layers[:,:,:,:,:]
    
    l1_mean = tf.reduce_sum(l1, [3,4])
    l1_mean = tf.div(l1_mean, int(N_x)*int(N_y))
    print(l1_mean.get_shape())
    l1_mean = tf.stack([l1_mean]*int(N_x), axis = 3)
    print(l1_mean.get_shape())

    l1_mean = tf.stack([l1_mean]*int(N_y), axis = 4)
    print(l1_mean.get_shape())
    
    steps = BS -1
    for i in range(1,steps):
        part1 = l1[:,0:int(N)-i,:,:,:]	
        part2 = l1[:,int(N)-i:,:,:,:]    
        shifted_layer = tf.concat([part2,part1],1)
        part1_mean = l1_mean[:,0:int(N)-i,:,:,:]	
        part2_mean = l1_mean[:,int(N)-i:,:,:,:]    
        shifted_mean = tf.concat( [part2_mean,part1_mean],1)     
        
        l11 = tf.subtract(l1, l1_mean)
        std = tf.square(l11)
        std = tf.reduce_sum(std, [2])
        std = tf.reduce_sum(std, [2,3])
        std = tf.stack([std]*int(dim), axis = 2)
        std = tf.stack([std]*int(N_x), axis = 3)
        std = tf.stack([std]*int(N_y), axis = 4)    
        std = tf.div(std, int(N_x)*int(N_y)-1)
        std = tf.sqrt(std)
        
        l22 = tf.subtract(shifted_layer, shifted_mean)
        std2 = tf.square(l22)
        std2 = tf.reduce_sum(std2, [2])
        std2 = tf.reduce_sum(std2, [2,3])
        std2 = tf.stack([std2]*int(dim), axis = 2)
        std2 = tf.stack([std2]*int(N_x), axis = 3)
        std2 = tf.stack([std2]*int(N_y), axis = 4)    
        std2 = tf.div(std2, int(N_x)*int(N_y)-1)
        std2 = tf.sqrt(std2)
        
        l11 = tf.divide(l11, std)
        l22 = tf.divide(l22, std2)
        p = tf.multiply(l11,l22)
        p = tf.reduce_sum(p, [2])
        p = tf.reduce_sum(p, [2,3])
        p = tf.div(p, int(N_x)*int(N_y)-1)
        p = tf.square(p) #square pearson
        p = tf.reduce_sum(p, 1, keep_dims = True)
        p = tf.div(p, int(N)-1)
        
        if i != 1:
            pearson = tf.concat([pearson, p],1)
        else:
            pearson = p
    pearson = tf.reduce_sum(pearson, 1, keep_dims = True)
    pearson = tf.div(pearson, steps -1)
            
    return pearson 

  
J = np.array([1.0,1.0,1.0])  
A = np.array([0.0,0.0,1.0])
def energy(layers):
    (N,dim,N_x,N_y) = layers.get_shape()
        
    l_x = layers[:,0,:,:]    
    l_y = layers[:,1,:,:]    
    l_z = layers[:,2,:,:]    

    l_x_a = tf.scalar_mul(A[0],l_x)
    l_y_a = tf.scalar_mul(A[1],l_y)
    l_z_a = tf.scalar_mul(A[2],l_z)
    
    layers_a = tf.stack([l_x_a,l_y_a,l_z_a], axis = 1)
    layers_a = tf.multiply(layers_a,layers_a)
    a = tf.reduce_sum(layers_a, [1,2,3])
    
    l_x = tf.scalar_mul(J[0],l_x)
    l_y = tf.scalar_mul(J[1],l_y)
    l_z = tf.scalar_mul(J[2],l_z)
    
    layers = tf.stack([l_x,l_y,l_z], axis = 1)
    
    p1 = layers[:,:,:,0:int(N_y)-1]	
    p2 = layers[:,:,:,int(N_y)-1:]    
    sl1 = tf.concat( [p2,p1], 3)  
     
    #p1 = layers[:,:,:,0:int(N_y)+1]	
    #p2 = layers[:,:,:,int(N_y)+1:]    
    #sl2 = tf.concat(3, [p2,p1])
    p1 = layers[:,:,0:int(N_x)-1,:]	
    p2 = layers[:,:,int(N_x)-1:,:]    
    sl3 = tf.concat( [p2,p1], 2)
    #p1 = layers[:,:,0:int(N_x)+1,:]	
    #p2 = layers[:,:,int(N_x)+1:,:]    
    #sl4 = tf.concat(2, [p2,p1])

    e1 = tf.multiply(layers, sl1)
    e1 = tf.reduce_sum(e1, [1,2,3])

    #e2 = tf.multiply(layers, sl2)
    #e2 = tf.reduce_sum(e2, [1,2,3])
    e3 = tf.multiply(layers, sl3)
    e3 = tf.reduce_sum(e3, [1,2,3])
    #e4 = tf.multiply(layers, sl4)
    #e4 = tf.reduce_sum(e4, [1,2,3])

    #e = e1
    e = tf.add(e1,e3)
    #e = tf.add(e,e3)
    #e = tf.add(e,e4)
    
    a = tf.scalar_mul(-1.0,a)
    e = tf.scalar_mul(-1.0,e)
    #result = e
    result = tf.add(a,e)
    result = tf.div(result, int(N_x)*int(N_y))
    return result
        
        
        
def mag_plus(layers):
    (N,dim,N_x,N_y) = layers.get_shape()
    layers = tf.reduce_sum(layers, [2,3])
    layers = tf.div(layers, int(N_x)*int(N_y))
    layers = tf.multiply(layers, layers)
    layers = tf.reduce_sum(layers, 1)
    layers = tf.sqrt(layers)
    return layers    
    energy
    
    
#########################################---------------------- Correlator---------------------################################################
get_corr_inp = Input(shape=(3,40, 40))
get_corr_out = Lambda(correlation, output_shape = corr_out_shape)(get_corr_inp)

get_corr = Model(get_corr_inp, get_corr_out)

get_corr.compile(loss ='mse', optimizer=opt)
get_corr.summary()


#########################################---------------------- Magnetization---------------------################################################
get_mag_inp = Input(shape=(3,40, 40))
get_mag_out = Lambda(mag_plus, output_shape = meanplus_shape)(get_mag_inp)

get_mag = Model(get_mag_inp, get_mag_out)

get_mag.compile(loss ='mse', optimizer=opt)
get_mag.summary()


#########################################---------------------- Energy---------------------################################################
get_energy_inp = Input(shape=(3,40, 40))
get_energy_out = Lambda(energy, output_shape = meanplus_shape)(get_energy_inp)

get_energy = Model(get_energy_inp, get_energy_out)

get_energy.compile(loss ='mse', optimizer=opt)
get_energy.summary()


#########################################---------------------- Correlator---------------------################################################
get_corr_enc_inp = Input(shape=(3,20, 20))
get_corr_enc_out = Lambda(correlation, output_shape = corr_out_shape)(get_corr_enc_inp)

get_corr_enc = Model(get_corr_enc_inp, get_corr_enc_out)

get_corr_enc.compile(loss ='mse', optimizer=opt)
get_corr_enc.summary()


#########################################---------------------- Magnetization---------------------################################################
get_mag_enc_inp = Input(shape=(3,20, 20))
get_mag_enc_out = Lambda(mag_plus, output_shape = meanplus_shape)(get_mag_enc_inp)

get_mag_enc = Model(get_mag_enc_inp, get_mag_enc_out)

get_mag_enc.compile(loss ='mse', optimizer=opt)
get_mag_enc.summary()


#########################################---------------------- Energy---------------------################################################
get_energy_enc_inp = Input(shape=(3,20, 20))
get_energy_enc_out = Lambda(energy, output_shape = meanplus_shape)(get_energy_enc_inp)

get_energy_enc = Model(get_energy_enc_inp, get_energy_enc_out)

get_energy_enc.compile(loss ='mse', optimizer=opt)
get_energy_enc.summary()



get_energy_batch_inp = Input(shape=(BS*n_feat_maps, 3,20, 20))

get_energy_batch_out = TimeDistributed(Lambda(energy,output_shape=meanplus_shape),input_shape=(BS*n_feat_maps, 3, 40, 40))(get_energy_batch_inp)

get_energy_batch = Model(get_energy_batch_inp, get_energy_batch_out)

get_energy_batch.compile(loss ='mse', optimizer=opt)
get_energy_batch.summary()



get_corr_batch_inp = Input(shape=(BS, 3,40, 40))

get_corr_batch_out = TimeDistributed(Lambda(correlation,output_shape=corr_out_shape),input_shape=(BS*n_feat_maps, 3, 40, 40))(get_corr_batch_inp)

get_corr_batch = Model(get_corr_batch_inp, get_corr_batch_out)

get_corr_batch.compile(loss ='mse', optimizer=opt)
get_corr_batch.summary()

#######################################################################



get_p_inp = Input(shape=(BS, 3,40, 40))

get_p_out = Lambda(pearson_corr)(get_p_inp)

get_p = Model(get_p_inp, get_p_out)

get_p.compile(loss ='mse', optimizer=opt)
get_p.summary()


