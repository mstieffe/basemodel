import os
import numpy as np
from keras.utils import generic_utils
from keras.utils import np_utils

from config import *
from models_stat import *

def increase_lr(n,n_max,lr,inc):
    if n < n_max:
        K.set_value(lr, K.get_value(lr) +inc)
    return lr
    
def increase_lr2(n,n_max,lr,inc):
    if n < n_max + 4 and n > 4:
        K.set_value(lr, K.get_value(lr) +inc)
    return lr

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
"""
def make_noise_sample(scale):
    noise_sample = scale * np.random.randn(BS, noise_dim)
    return noise_sample
"""

def make_batch_for_dis(X_train, y_train, encoder, decoder,num_classes):
    idx = np.random.choice(np.arange(X_train.shape[0]), BS*n_feat_maps, replace=False)
    X_sample = X_train[idx]
    y_sample = y_train[idx]
    
    idx = np.random.choice(np.arange(X_train.shape[0]),BS*n_feat_maps, replace=False)
    train_img = X_train[idx]   
    train_img = encoder.predict(train_img)

    if noise_dim == 0:
        #print(train_img.shape)
        decode_img = decoder.predict(train_img)
    else:
        noise_sample = make_noise_sample(noise_scale, BS)
        decode_img = decoder.predict([train_img, noise_sample])

    X_sample = np.concatenate((X_sample, decode_img),axis = 0)
      
    y = np.array(BS*n_feat_maps*[num_classes-1])
    y = np_utils.to_categorical(y) 
    y_sample = np.concatenate((y_sample,y))    
    
    return X_sample, y_sample
     
def make_batch_for_dis_noise(X_train, y_train, encoder, decoder,num_classes): 
    idx = np.random.choice(np.arange(X_train.shape[0]),BS, replace=False)
    train_img = X_train[idx]   
    train_img = encoder.predict(train_img)

    noise_sample = make_noise_sample(noise_scale, BS)
    decode_img = decoder.predict([train_img, noise_sample])
    
    return decode_img, noise_sample    

    
def make_batch_for_gan(X_train, y_train,stat_samples,encoder, num_classes):
    
    
    
    
    if pearson_loss:
        #chose random samples with the same temperature
        temp = np.random.randint(num_classes-1)
        idx = np.random.choice(np.arange(stat_samples[temp].shape[0]), 1, replace=False)
        idx = list(idx) * BS
        real_samples = stat_samples[temp][idx]
        
        
        #target values for temperature discriminator
        y_temp = np_utils.to_categorical([temp]*BS*n_feat_maps, num_classes = num_classes)
        y_temp = np.reshape(y_temp, (1,BS*n_feat_maps, num_classes)) 
        
    else:
        #chose random samples with random temperature
        idx = np.random.choice(np.arange(X_train.shape[0]), BS, replace=False)
        real_samples = X_train[idx]
        
        #y_temp = y_train[idx]        
        #y_temp = np.reshape(y_temp, (1,BS, num_classes))         
 
        y_temp = np.repeat(y_train[[idx[0]],:], n_feat_maps, axis = 0)
        for n in range(1,BS):        
            y_temp = np.concatenate((y_temp,np.repeat(y_train[[idx[n]],:], n_feat_maps, axis = 0)), axis = 0)
        y_temp = np.reshape(y_temp, (1,BS*n_feat_maps,num_classes))

       
    enc_samples = encoder.predict(real_samples)
    
    X_sample = [np.reshape(enc_samples, (1,BS,3,20,20))]
    
    if noise_dim != 0:
        noise_sample = noise_scale * np.random.randn(1,BS, noise_dim)
        X_sample.append(noise_sample)

    if spec_h_loss:
        #batch of temperature values for the statistic learning
        temps = np.array([[temp_list[temp]]*(BS*n_feat_maps)])
        X_sample.append(temps)



    y_sample = [y_temp]



    #target values for reconstruction error
    if recon:
        y_recon = np.repeat(enc_samples[[0],:,:,:], n_feat_maps, axis = 0)
        for n in range(1,BS):        
            y_recon = np.concatenate((y_recon,np.repeat(enc_samples[[n],:,:,:], n_feat_maps, axis = 0)), axis = 0)
        y_recon = np.reshape(y_recon, (1,BS*n_feat_maps,3,20,20))
        y_sample.append(y_recon)
    if pearson_loss:    
        y_sample.append(np.array([[0.0]]))
    
    if noise_dis:
        noise_sample = noise_sample[0,:,:]
        y_noise = np.repeat(noise_sample[[0],:], n_feat_maps, axis = 0)
        for n in range(1,BS):        
            y_noise = np.concatenate((y_noise,np.repeat(noise_sample[[n],:], n_feat_maps, axis = 0)), axis = 0)
        y_noise = np.reshape(y_noise, (1,BS*n_feat_maps,noise_dim))        
        y_sample.append(y_noise)


    return X_sample, y_sample
    #return X_sample, [y_temp,y_recon, np.reshape(stat_list[0][temp], (1,64)), np.reshape(stat_list[6][temp], (1,1)), np.reshape(stat_list[2][temp], (1,1)), np.reshape(stat_list[3][temp], (1,1)), np.reshape(stat_list[4][temp], (1,1))]


def pre_training(X_train, Y_train,encoder, decoder, dis_temp,num_classes,steps = 10):
    progbar = generic_utils.Progbar(steps*BS)
    for n in range(0,steps):      
        X_sample, y_sample = make_batch_for_dis(X_train, Y_train ,encoder, decoder,num_classes)        
        
        #unfreeze the weights of the discriminator
        make_trainable(dis_temp,True)   
        #train the discriminator
        dt_loss = dis_temp.train_on_batch(X_sample, y_sample)
        
        progbar.add(BS, values=[("Loss_D", dt_loss[0]),
                                ("Acc_D", dt_loss[1])])         
      
 		
def training(losses,X_train, Y_train,stat_samples, encoder, decoder, dis_temp,dis_noise, dis,gan,num_classes,nb_epoch=5000):
    #display the progess of the learning process    
    progbar = generic_utils.Progbar(nb_epoch*BS)
    for e in range(0,nb_epoch):
    #for e in tqdm(range(nb_epoch)):  

        X_sample, y_sample = make_batch_for_dis(X_train, Y_train,encoder, decoder,num_classes)
        #make_trainable(dis,True) 
        make_trainable(dis_temp,True) 
        dt_loss  = dis_temp.train_on_batch(X_sample,y_sample)
        losses["dt"].append(dt_loss)

        if noise_dis:
            X_sample, y_sample = make_batch_for_dis_noise(X_train, Y_train,encoder, decoder,num_classes)
    
            make_trainable(dis_noise,True) 
            make_trainable(dis,True) 
            dn_loss  = dis_noise.train_on_batch(X_sample,y_sample)
            losses["dn"].append(dn_loss)  
            make_trainable(dis_noise,False) 
            make_trainable(dis,False) 

            
        #####################################################
        
        X_sample, y_target_list = make_batch_for_gan(X_train, Y_train,stat_samples,encoder, num_classes)
       
        make_trainable(dis_temp,False)
        g_loss = gan.train_on_batch(X_sample,y_target_list)
        #g_loss = [0]
        losses["g"].append(g_loss)

        prog_list = [("DT", dt_loss[0]), ("Acc", dt_loss[1])]
        
        if noise_dis:
            prog_list.append(("DN", dn_loss[0]))
            prog_list.append(("A", dn_loss[1]))


        prog_list.append(("G", g_loss[0]))  
        for n in range(1,len(g_loss)):
            prog_list.append((str(n), g_loss[n]))   
        
        progbar.add(BS, values= prog_list)

    return losses
        
