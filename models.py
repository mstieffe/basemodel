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
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D, Conv2DTranspose, Convolution3D
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
from keras.layers.pooling import GlobalAveragePooling2D
K.set_image_dim_ordering('th')
K.set_learning_phase(1)


from config import *
from models_stat import *




def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def normalize(layers):
    norm = tf.square(layers)
    norm = tf.reduce_sum(norm, 1, keep_dims=True)
    norm = tf.sqrt(norm) 
    norm = tf.concat([norm,norm, norm], 1)
    layers= tf.div(layers, norm)

    return layers


def normalize_set(layers):
    norm = tf.square(layers)
    norm = tf.reduce_sum(norm, 2, keep_dims=True)
    norm = tf.sqrt(norm) 
    norm = tf.concat([norm,norm, norm], 2)
    layers= tf.div(layers, norm)

    return layers

def random_feat_map(layers):
    part = tf.random_crop(layers, [BS,1,nb_ch_dec,20,20])
    return part    
    
def random_feat_map2(layers):
    part = tf.random_crop(layers, [BS,1,nb_ch_dec/2,40,40])
    return part    
    
def randfeat_out_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],1,shape[2],shape[3],shape[4]]) 
 
        
    
def random_layers(layers):
    print(layers.get_shape())
    part = tf.random_crop(layers, [32,1,3,40,40])
    print(part.get_shape())
    return part    
    
def randlayer_out_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],1,shape[2],shape[3],shape[4]]) 
    
    
def combine_axis(layers):
    new_layers = layers[:,0,:,:,:,:]
    for n in range(1,BS):
        new_layers = tf.concat([new_layers, layers[:,n,:,:,:,:]], 1)
    return new_layers
    
def combine_axis_out_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],shape[1]*shape[2],shape[3],shape[4],shape[5]]) 

def meanplus_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],1])

def ave_out_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],shape[2]])

    
def averager(layers):
    layers = tf.reduce_sum(layers, [1])
    layers = tf.div(layers, BS*n_feat_maps)
    return layers

############################################# Encoder Model #######################################
def make_encoder():
    encoder_inp = Input(shape=(3,None,None))
    H = AveragePooling2D(pool_size=(2, 2))(encoder_inp)
    encoder_out = Lambda(normalize)(H)
    
    encoder = Model(encoder_inp, encoder_out)
    encoder.summary()
    return encoder


#############################################  Decoder Model ######################################
def make_decoder(name):
    decoder_inp = Input(shape=(3,20,20))
    H = Conv2D(nb_ch_dec,(4,4), padding='same', init='glorot_uniform', name='decoder1')(decoder_inp)
    H = Activation('tanh')(H)
    H = Conv2DTranspose(nb_ch_dec/2*n_feat_maps, (4,4), output_shape=(BS,nb_ch_dec/2*n_feat_maps, 40, 40), strides=(2, 2), padding='same', name='decoder2')(H)
    H = Activation('tanh')(H)
    H = Reshape((n_feat_maps,nb_ch_dec/2,40,40))(H)   
    H = TimeDistributed(Conv2D(3,(4,4), padding='same', init='glorot_uniform', name='decoder3'),input_shape=(n_feat_maps, nb_ch_dec/2, 40, 40))(H)
    H = Activation('tanh')(H)    
    decoder_out = Lambda(normalize_set)(H)
    
    decoder = Model(decoder_inp, decoder_out)
    decoder.summary()
    return decoder


  
def make_decoder_dense(name):
    decoder_inp = Input(shape=(3,20,20))
    H = Flatten()(decoder_inp)
    
    H = Dense(nb_ch_dec*20*20, name='dec_noise3')(H)
    H = Activation('tanh')(H)
    H = Reshape((nb_ch_dec,20,20), input_shape=(nb_ch_dec*20*20,))(H)    
    
    #H = Conv2D(nb_ch_dec,4,4, padding='same', init='glorot_uniform', name='decoder1')(decoder_inp)
    #H = Activation('tanh')(H)
    H = Conv2DTranspose(nb_ch_dec/2*n_feat_maps, (4,4), output_shape=(BS,nb_ch_dec/2*n_feat_maps, 40, 40), strides=(2, 2), padding='same', name='decoder2')(H)
    H = Activation('tanh')(H)
    H = Reshape((n_feat_maps,nb_ch_dec/2,40,40))(H)   
    H = TimeDistributed(Conv2D(3,(4,4), padding='same', init='glorot_uniform', name='decoder3'),input_shape=(n_feat_maps, nb_ch_dec/2, 40, 40))(H)
    H = Activation('tanh')(H)    
    decoder_out = Lambda(normalize_set)(H)
    
    decoder = Model(decoder_inp, decoder_out)
    decoder.summary()
    return decoder
    

def make_dec_noise():
    dec_noise_inp = Input(shape=(noise_dim,))
    B = Dense(128, name='dec_noise1')(dec_noise_inp)
    B = Activation('tanh')(B)
    B = Dense(128, name='dec_noise2')(B)
    dec_noise_out = Activation('tanh')(B)

    dec_noise = Model(dec_noise_inp, dec_noise_out)
    dec_noise.summary()
    return dec_noise

def make_dec_noise2():
    dec_noise_inp2 = Input(shape=(128+3*20*20,))
    C = Dense(nb_ch_dec*20*20, name='dec_noise3')(dec_noise_inp2)
    C = Activation('tanh')(C)
    C = Reshape((nb_ch_dec,20,20), input_shape=(nb_ch_dec*20*20,))(C)
    
    C = Conv2DTranspose(nb_ch_dec/2*n_feat_maps, (4,4), output_shape=(BS,nb_ch_dec/2*n_feat_maps, 40, 40), strides=(2, 2), padding='same', name='decoder2')(C)
    C = Activation('tanh')(C)
    C = Reshape((n_feat_maps,nb_ch_dec/2,40,40))(C)   
    C = TimeDistributed(Conv2D(3,(4,4), padding='same', init='glorot_uniform', name='decoder3'),input_shape=(n_feat_maps, nb_ch_dec/2, 40, 40))(C)
    C = Activation('tanh')(C)    
    dec_noise_out2 = Lambda(normalize_set)(C)
    
    dec_noise2 = Model(dec_noise_inp2, dec_noise_out2)
    dec_noise2.summary()
    return dec_noise2
 
def make_decoder_noise():
    dec_noise = make_dec_noise()
    dec_noise2 = make_dec_noise2()
  
    make_trainable(dec_noise,True)
    make_trainable(dec_noise2,True)

  
    decoder_inp1 = Input(shape=(3,20,20))
    decoder_inp2 = Input(shape=(noise_dim,))
    
    B = dec_noise(decoder_inp2)    
    
    H = Flatten()(decoder_inp1)
    C = merge([H, B], mode='concat', concat_axis= 1)
    
    decoder_out = dec_noise2(C)

    decoder_noise = Model([decoder_inp1,decoder_inp2], decoder_out)
    decoder_noise.summary()
    return dec_noise, dec_noise2, decoder_noise

def make_dec_sing(decoder):
    if noise_dim == 0:
        dec_sing_inp = Input(shape=(3,20,20))
        H = decoder(dec_sing_inp)
        H = Lambda(random_layers,output_shape=randlayer_out_shape)(H)
        dec_sing_out = Reshape((3,40,40))(H)
            
        dec_sing = Model(dec_sing_inp, dec_sing_out)
        dec_sing.summary()
        return dec_sing        
    else:
        dec_sing_inp = Input(shape=(3,20,20))
        dec_sing_inp2 = Input(shape=(noise_dim,))
        H = decoder([dec_sing_inp,dec_sing_inp2])
        H = Lambda(random_layers,output_shape=randlayer_out_shape)(H)
        dec_sing_out = Reshape((3,40,40))(H)
            
        dec_sing = Model([dec_sing_inp,dec_sing_inp2], dec_sing_out)
        dec_sing.summary()
        return dec_sing
    

#########################################----------------------DISCRIMINATOR - TEMPERATURE---------------------################################################
	
def make_dis_temp(num_classes):    
    #define input tenso
    dis_temp_inp = Input(shape=(3,40,40))
    
    #convolution layer with 256 filter of size 5x5x1 resulting in 256 feature maps of size 28x28
    H = Conv2D(nb_ch_dis,(4,4), strides=(2,2), padding ='same')(dis_temp_inp)
    # leakyRelu activation function: x>0: max(0,x), x<0: alpha*x
    H = LeakyReLU(0.2)(H)
    #dropout
    H = Dropout(dr_dis)(H)
    #convolution layer with 512 filter of size 5x5x256 resulting in 512 feature maps of size 28x28
    H = Conv2D(nb_ch_dis*2,(4,4), strides=(2,2), padding ='same')(H)
    # leakyRelu activation function: x>0: max(0,x), x<0: alpha*x
    H = LeakyReLU(0.2)(H)
    #dropout
    H = Dropout(dr_dis)(H)
    #make a vector out of the 3d array
    H = Flatten()(H)
    #fully connected layer 
    H = Dense(256)(H)
    #leaky relu
    H = LeakyReLU(0.2)(H)
    #dropout
    H = Dropout(dr_dis)(H)
    #fully connected layer with 2 neurons for output
    dis_temp_out = Dense(num_classes, activation='softmax')(H)	
    	
    dis_temp = Model(dis_temp_inp, dis_temp_out)
    #define learning rule
    dis_temp.summary()	
    return dis_temp
    
    
def make_dis_temp_and_noise(num_classes):

    dis_inp = Input(shape=(3,40,40))

    H = Conv2D(nb_ch_dis,(4,4), strides=(2,2), padding ='same')(dis_inp)
    # leakyRelu activation function: x>0: max(0,x), x<0: alpha*x
    H = LeakyReLU(0.2)(H)
    #dropout
    H = Dropout(dr_dis)(H)
    #convolution layer with 512 filter of size 5x5x256 resulting in 512 feature maps of size 28x28
    H = Conv2D(nb_ch_dis*2,(4,4), strides=(2,2), padding ='same')(H)
    # leakyRelu activation function: x>0: max(0,x), x<0: alpha*x
    H = LeakyReLU(0.2)(H)
    dis_out = Dropout(dr_dis)(H)


    dis = Model(dis_inp, dis_out)
    #define learning rule
    dis.compile(loss='categorical_crossentropy', optimizer=dis_opt, metrics=['accuracy'])
    dis.summary()	
    
    
    make_trainable(dis,True)
    ################# temp dis #############################
    dis_temp_inp = Input(shape=(3,40,40))
    
    H = dis(dis_temp_inp)
    H = Flatten()(H)
    #fully connected layer 
    H = Dense(256)(H)
    #leaky relu
    H = LeakyReLU(0.2)(H)
    H = Dropout(dr_dis)(H)

    dis_temp_out = Dense(num_classes, activation='softmax')(H)
    
    	
    dis_temp = Model(dis_temp_inp, dis_temp_out)
    #define learning rule
    dis_temp.summary()	
    dis_temp.compile(loss='categorical_crossentropy', optimizer=dis_opt, metrics=['accuracy'])
    
    
    make_trainable(dis,True)
    ####################### noise dis #############
    dis_noise_inp = Input(shape=(3,40,40))
    """
    H = dis(dis_noise_inp)
    
    H = Conv2D(noise_dim,3,3, strides=(2,2), padding ='same')(H)
    H = GlobalAveragePooling2D()(H)
    
    
    dis_noise_out =  Activation('tanh')(H)
    """
    
    H = dis(dis_noise_inp)
    H = Flatten()(H)
    #fully connected layer 
    H = Dense(256)(H)
    #leaky relu
    H = LeakyReLU(0.2)(H)
    dis_noise_out = Dense(noise_dim, activation='tanh')(H)

    	
    dis_noise = Model(dis_noise_inp, dis_noise_out)
    #define learning rule
    dis_noise.summary()

    dis_noise.compile(loss='mse', optimizer=dis_opt, metrics=['accuracy'])

    return dis_temp, dis_noise, dis
#########################################----------------------GAN - Temperature + stat corr ---------------------################################################


def make_gan(decoder, dis_temp, encoder, dis_noise):
    gan_inp1 = Input(shape=(BS,3,20,20))
    
    gan_inp = [gan_inp1]    
    gan_out = []
    
    H = TimeDistributed(decoder,input_shape=(BS, 3, 40, 40))(gan_inp1)    
    H = Lambda(combine_axis,output_shape=combine_axis_out_shape)(H)
    
    #### adversarial loss from temperature discriminator
    gan_out.append(TimeDistributed(dis_temp,input_shape=(BS * n_feat_maps, 3, 40, 40))(H))
    
    ### reconstruction loss
    if recon:
        R = TimeDistributed(encoder,input_shape=(BS * n_feat_maps, 3, 40, 40))(H)
        gan_out.append(R)
        
    #Pearson loss
    if pearson_loss:
        P = Lambda(pearson_corr)(H)
        gan_out.append(P)
        
    gan = Model(gan_inp, gan_out)
    gan.summary()
    return gan
    
def make_gan_noise(dec_noise, dec_noise2, encoder, dis_temp, dis_noise):
    gan_inp1 = Input(shape=(BS,3,20,20))
    gan_inp2 = Input(shape=(BS, noise_dim))
    gan_inp3 = Input(shape=(1,))
    
    gan_inp = [gan_inp1,gan_inp2]    

    if spec_h_loss:
        gan_inp.append(gan_inp3)
        
    
    gan_out = []   
    
    B = TimeDistributed(dec_noise,input_shape=(BS, noise_dim))(gan_inp2) 
    
    C = TimeDistributed(Flatten(),input_shape=(BS, 3,20,20))(gan_inp1)     
    
    H = merge([C, B], mode='concat', concat_axis= 2)
    
    H = TimeDistributed(dec_noise2,input_shape=(BS, 128+3*20*20))(H) 
    H = Lambda(combine_axis,output_shape=combine_axis_out_shape)(H)   

    gan_out.append(TimeDistributed(dis_temp,input_shape=(BS * n_feat_maps, 3, 40, 40))(H))
    
    ### reconstruction loss
    if recon:
        R = TimeDistributed(encoder,input_shape=(BS * n_feat_maps, 3, 40, 40))(H)
        gan_out.append(R)
        
    #noise discrimination error
    if pearson_loss:
        P = Lambda(pearson_corr)(H)
        gan_out.append(P)
    if noise_dis:
        ND = TimeDistributed(dis_noise,input_shape=(BS * n_feat_maps, 3, 40, 40))(H)
        gan_out.append(ND)
  
    
    gan = Model(gan_inp, gan_out)
    gan.summary()
    return gan