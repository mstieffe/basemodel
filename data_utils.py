import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from models_stat import *


K.set_image_dim_ordering('th')

############################ Load data and preprocessing #######################


def load_data():
    print("start loading data")
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.1c.pickle', 'rb')
    x_train1 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.2c.pickle', 'rb')
    x_train2 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.3c.pickle', 'rb')
    x_train3 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.4c.pickle', 'rb')
    x_train4 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.5c.pickle', 'rb')
    x_train5 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.6c.pickle', 'rb')
    x_train6 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.7.pickle', 'rb')
    x_train7 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.8.pickle', 'rb')
    x_train8 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.9.pickle', 'rb')
    x_train9 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.0.pickle', 'rb')
    x_train10 = pickle.load(file)
    file.close()
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.04.pickle', 'rb')
    x_train104 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.1.pickle', 'rb')
    x_train11 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.2.pickle', 'rb')
    x_train12 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.3.pickle', 'rb')
    x_train13 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.4.pickle', 'rb')
    x_train14 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.5.pickle', 'rb')
    x_train15 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.6.pickle', 'rb')
    x_train16 = pickle.load(file)
    file.close() 
    """
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.7.pickle', 'rb')
    x_train17 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.8.pickle', 'rb')
    x_train18 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.9.pickle', 'rb')
    x_train19 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T2.0.pickle', 'rb')
    x_train20 = pickle.load(file)
    file.close() 
    """
    print("loading data successfull")
    
    x_train = np.concatenate((x_train1, x_train2), axis = 0)
    x_train = np.concatenate((x_train, x_train3), axis = 0)
    x_train = np.concatenate((x_train, x_train4), axis = 0)
    x_train = np.concatenate((x_train, x_train5), axis = 0)
    x_train = np.concatenate((x_train, x_train6), axis = 0)
    x_train = np.concatenate((x_train, x_train7), axis = 0)
    x_train = np.concatenate((x_train, x_train8), axis = 0)
    x_train = np.concatenate((x_train, x_train9), axis = 0)
    x_train = np.concatenate((x_train, x_train10), axis = 0)
    x_train = np.concatenate((x_train, x_train104), axis = 0)
    x_train = np.concatenate((x_train, x_train11), axis = 0)
    x_train = np.concatenate((x_train, x_train12), axis = 0)
    x_train = np.concatenate((x_train, x_train13), axis = 0)
    x_train = np.concatenate((x_train, x_train14), axis = 0)
    x_train = np.concatenate((x_train, x_train15), axis = 0)
    x_train = np.concatenate((x_train, x_train16), axis = 0)

    stat_samples = [x_train1, x_train2 , x_train3, x_train4, x_train5, x_train6, x_train7, x_train8, x_train9, x_train10, x_train104, x_train11, x_train12, x_train13, x_train14, x_train15, x_train16]

    num_classes = len(stat_samples)+1

    y_train = np.array(x_train1.shape[0]*[0]+x_train2.shape[0]*[1]+x_train3.shape[0]*[2]+x_train4.shape[0]*[3]+x_train5.shape[0]*[4]+x_train6.shape[0]*[5]+x_train7.shape[0]*[6]+5000*[7]+5000*[8]+5000*[9]+5000*[10]+5000*[11]+5000*[12]+5000*[13]+5000*[14]+5000*[15]+5000*[16])
    y_train = np_utils.to_categorical(y_train, num_classes = num_classes)    

    temps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.04,1.1,1.2,1.3,1.4,1.5,1.6]  

    return x_train, y_train, num_classes, stat_samples, temps



def ave_out_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],shape[2]])

    
def averager(layers):
    layers = tf.reduce_sum(layers, [1])
    layers = tf.div(layers, 500)
    return layers
    
def get_stats(data_set, num_samples = 1000):
    correlation_list = []
    mag_list = []
    mag_sq_list = []
    en_list = []
    en_sq_list = []
    

    #print('Shape of input data: '+str(data_set[0].shape))
    for data in data_set:
        idx = np.random.choice(np.arange(data.shape[0]), num_samples, replace=False)
        data = data[idx]
        (N,dim, N_x, N_y) = data.shape
        
        if N_x == 40:
            corr = get_corr.predict(data, batch_size=N)
            mp = get_mag.predict(data, batch_size=N)
            en = get_energy.predict(data, batch_size=N)
        else:
            corr = get_corr_enc.predict(data, batch_size=N)
            mp = get_mag_enc.predict(data, batch_size=N)
            en = get_energy_enc.predict(data, batch_size=N)
            
        corr = np.sum(corr, axis = 0)
        corr = corr / N		
        corr = list(corr)    
    
        mp_sq = mp[:]
        mp_sq = np.square(mp_sq)     
        mp_sq = np.sum(mp_sq)
        mp_sq = mp_sq / N
        
        mp = np.sum(mp)
        mp = mp/ N
        
        en_sq = en[:]
        en_sq = np.square(en_sq)
        en_sq = np.sum(en_sq)
        en_sq = en_sq / N
    
        
        en = np.sum(en)
        en= en / N
        

    
        correlation_list.append(corr)
        mag_list.append(mp)
        en_list.append(en)
        en_sq_list.append(en_sq)
        mag_sq_list.append(mp_sq)  
        
    temps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.04,1.1,1.2,1.3,1.4,1.5, 1.6]
    
    sus_list = []
    for n in range(0,len(mag_list)):
        sus_list.append((mag_sq_list[n] - mag_list[n]*mag_list[n])/temps[n])

        
    spez_list = []
    for n in range(0,len(en_list)):
        spez_list.append((en_sq_list[n] - en_list[n]*en_list[n])/(temps[n]*temps[n]))
    
    
    return [correlation_list, en_list, en_sq_list , mag_list ,mag_sq_list, sus_list, spez_list]

def samples_for_randfeat_plot(real_samples, enc_samples):
    real_img_rf = []
    for samples in real_samples:
        real_img_rf.append(np.repeat(samples[[0],:,:,:], n_feat_maps, axis = 0))
    enc_img_rf = []
    for samples in enc_samples:
        enc_img_rf.append(np.repeat(samples[[0],:,:,:], n_feat_maps, axis = 0))    
    return real_img_rf,enc_img_rf

def get_samples_noise(stat_samples, encoder, decoder, dis_noise):
    idx = np.random.choice(np.arange(stat_samples[0].shape[0]), BS, replace=False)
    samples = stat_samples[0][idx]
    samples = np.repeat(samples[[0],:,:,:], BS, axis = 0)
    real_samples = [samples]
    enc_samples = [encoder.predict(real_samples)]

    noise_sample = make_noise_sample(noise_scale)
    noise_samples = [noise_sample]
    dec_samples = [decoder.predict([enc_samples[-1], noise_sample]) ] 
    if noise_dis:
        dec_noise_samples =  [dis_noise.predict(dec_samples[-1][:,0,:,:,:])]       
    else:
        dec_noise_samples = []
        
    
    for n in range(1, len(stat_samples)):
        idx = np.random.choice(np.arange(stat_samples[n].shape[0]), BS, replace=False)
        samples = stat_samples[n][idx]
        samples = np.repeat(samples[[0],:,:,:], BS, axis = 0)
        real_samples.append(samples)
        enc_samples.append(encoder.predict(real_samples[-1]))

        noise_sample = make_noise_sample(noise_scale)
        noise_samples.append(noise_sample)
        dec_samples.append(decoder.predict([enc_samples[-1], noise_sample]))  
        if noise_dis:
            dec_noise_samples.append(dis_noise.predict(dec_samples[-1][:,0,:,:,:]))

    return real_samples, enc_samples, dec_samples, noise_samples, dec_noise_samples


def get_samples_stats(stat_samples, encoder, decoder): 
    idx = np.random.choice(np.arange(stat_samples[0].shape[0]), BS, replace=False)
    real_samples = [stat_samples[0][idx]]
    enc_samples = [encoder.predict(real_samples)]

    if noise_dim != 0:
        noise_sample = make_noise_sample(noise_scale)
        dec_samples = [decoder.predict([enc_samples[-1], noise_sample]) ]        
    else:
        dec_samples = [decoder.predict([enc_samples[-1]])]     
        
    for n in range(1, len(stat_samples)):
        idx = np.random.choice(np.arange(stat_samples[n].shape[0]), BS, replace=False)
        real_samples.append(stat_samples[n][idx])

        enc_samples.append(encoder.predict(real_samples[-1]))
        if noise_dim != 0:
            noise_sample = make_noise_sample(noise_scale)
            dec_samples.append(decoder.predict([enc_samples[-1], noise_sample]))  
        else:
            dec_samples.append(decoder.predict([enc_samples[-1]]))
    return real_samples, enc_samples, dec_samples
    
def seperate_decoded_samples(dec_samples):
    dec_img_rf = []
    dec_img_batch = []
    dec_img_all = []
    for samples in dec_samples:
        dec_img_rf.append(samples[0,:,:,:,:])
        dec_img_batch.append(samples[:,0,:,:,:])
        dec_all = samples[0,:,:,:,:]
        for j in range(1,BS):
            dec_all = np.concatenate((dec_all, samples[j,:,:,:,:]), axis = 0)
        dec_img_all.append(dec_all)
    #print(dec_img_rf[0].shape)
    #print(dec_img_batch[0].shape)
    #print(dec_img_all[0].shape)

    return dec_img_rf, dec_img_batch, dec_img_all


def clip_correlation_range(correlation_list, max_dis= 10):
    #print(len(correlation_list))
    #print(len(correlation_list[0]))
    corr_list = []
    for n in range(0,len(correlation_list)):
        corr = []
        d = []
        dist = dist_corr(8)
        for i in range(0,len(dist)):
            if dist[i] >= 0.0  and dist[i]<=max_dis:
                d.append(dist[i])
                corr.append(correlation_list[n][i])    
        corr_list.append(corr)
    return corr_list, d

def submag_from_correlation(correlation_list, mag_list):
    c = correlation_list[:]
    for i in range(0,len(c)):
        c[i] = [(j-mag_list[i]*mag_list[i]) for j in c[i]] 
    return c
    
def take_some(n,b, samples1, samples2, samples3):
    idx = np.random.randint(0,n,b)	
    samples_taken1 = samples1[0][idx]
    samples_taken2 = samples2[0][idx]	
    samples_taken3 = samples3[0][idx]	
    for i in range(1,len(samples1)):
        idx = np.random.randint(0,n,b)	
        take1 = samples1[i][idx]
        take2 = samples2[i][idx]
        take3 = samples3[i][idx]
        samples_taken1 = np.concatenate((samples_taken1, take1),axis = 0) 
        samples_taken2 = np.concatenate((samples_taken2, take2),axis = 0) 
        samples_taken3 = np.concatenate((samples_taken3, take3),axis = 0) 


    return samples_taken1,samples_taken2,samples_taken3

def convert_samples_for_plot(samples):
    n=samples.shape[0]
    N_x = samples.shape[2]
    N_y = samples.shape[3]
    N_z = 1
    
    #make a vector plot of the spins            
    UU,VV,WW = [],[],[]
    
    for i in range(0, n):
        XX,YY,ZZ,U,V,W = [],[],[],[],[],[]
        for x in range(0,N_x):
            for y in range(0,N_y):
                for z in range(0,N_z):    
                    XX.append(x)
                    YY.append(y)
                    ZZ.append(z)
                    U.append(samples[i][0][x][y])
                    V.append(samples[i][1][x][y])
                    W.append(samples[i][2][x][y])
        UU.append(U)
        VV.append(V)
        WW.append(W)
    return XX,YY,ZZ,UU,VV,WW, N_x

def plot_noise(noise_samples, dec_noise_samples, num_temps,folder_name):

    idx = np.random.choice(np.arange(noise_samples.shape[1]), 9, replace=False)

    #print(noise_samples.shape)
    #print(dec_noise_samples.shape)

    noise_samp_sing = noise_samples[range(0,17),:,:]
    dec_noise_samp_sing = dec_noise_samples[range(0,17),:,:]
    
    noise_samp_sing = noise_samp_sing[:,idx,:]
    dec_noise_samp_sing = dec_noise_samp_sing[:,idx,:]
    #print(noise_samp_sing.shape)
    #print(dec_noise_samp_sing.shape)

    plt.figure(figsize=(25, 15))
    plt.suptitle("noise histogram", fontsize=26)
    ax = plt.subplot(1, 2,  1)
    plt.title("input noise", fontsize=24)
    plt.hist(np.reshape(noise_samples, (BS*noise_dim*n_batch_for_noise_plot*17)), bins='auto')
    plt.xlim([-1.0,1.0])

    
    ax = plt.subplot(1, 2, 2)
    plt.title("decoded noise", fontsize=24)
    plt.hist(np.reshape(dec_noise_samples, (BS*noise_dim*n_batch_for_noise_plot*17)), bins='auto')
    plt.xlim([-1.0,1.0])

    plt.savefig(model_name+"/"+folder_name+"/noise_hist.png")
    plt.close()

 
    noise_x = range(0,noise_dim)
    noise_x2 = [n + 0.4 for n in noise_x]
    labels = ["T = 0.1", "T = 0.2", "T = 0.3","T = 0.4","T = 0.5","T = 0.6","T = 0.7","T = 0.8","T = 0.9","T = 1.0","T = 1.04","T = 1.1","T = 1.2","T = 1.3","T = 1.4","T = 1.5","T = 1.6","T = 1.7","T = 1.8","T = 1.9","T = 2.0"]
    for t in range(0,noise_samp_sing.shape[0]):
        plt.figure(figsize=(35, 20))
        plt.suptitle(labels[t], fontsize=26)    
        
        for j in range(0,noise_samp_sing.shape[1]):
            ax = plt.subplot(3, 3,  j+1)    
    
            plt.bar(noise_x,noise_samp_sing[t,j,:],width = 0.4, color = "blue", label="input noise")
            #plt.plot(noise_sample2[0], color = "red")
        
            plt.bar(noise_x2,dec_noise_samp_sing[t,j,:],width = 0.4, color = "red", label="decoded noise")
            if noise_dim < 40:
                plt.xlim([0,noise_dim])            
            else:
                plt.xlim([0,40])
            plt.ylim([-1.0,1.0])
            if j == 0:
                plt.legend() 
        #plt.show()
        plt.savefig(model_name+"/"+folder_name+"/"+str(t)+".png")
        plt.close()
            


def plot_samples(n,b,real_samples, enc_samples, dec_samples,folder_name):
    real_samples, enc_samples, dec_samples = take_some(n,b, real_samples, enc_samples, dec_samples)

    XX,YY,ZZ,UU,VV,WW, N_x = convert_samples_for_plot(real_samples)
    XX2,YY2,ZZ2,UU2,VV2,WW2, N_x2 = convert_samples_for_plot(enc_samples)
    XX3,YY3,ZZ3,UU3,VV3,WW3, N_x3 = convert_samples_for_plot(dec_samples)
    
    cm = plt.cm.get_cmap('gray')

    r = 3
    labels = ["T = 0.1", "T = 0.2", "T = 0.3","T = 0.4","T = 0.5","T = 0.6","T = 0.7","T = 0.8","T = 0.9","T = 1.0","T = 1.04","T = 1.1","T = 1.2","T = 1.3","T = 1.4","T = 1.5","T = 1.6","T = 1.7","T = 1.8","T = 1.9","T = 2.0"]
    l = 0
    for k in range(0,len(WW),3):
        plt.figure(figsize=(35, 20))
        plt.suptitle(labels[l], fontsize=26)
        for i in range(3):
            ax = plt.subplot(3, r,  i*r +1)
            if i == 0:
                plt.title("real", fontsize=24)
            #plt.quiver(XX,YY,UU[i+k],VV[i+k],pivot = 'middle', color='red', scale =1, units='xy')
            plt.scatter(XX, YY, c=WW[i+k], vmin=-1.0, vmax=1.0, s=40, cmap=cm, edgecolors='none')
            plt.colorbar(ticks=[-1, 0, 1])
            plt.xlim([-1,N_x])
            plt.ylim([-1,N_x])
            
            ax = plt.subplot(3, r, 2+ i*r)
            if i == 0:
                plt.title("renorm", fontsize=24)
            #plt.quiver(XX2,YY2,UU2[i+k],VV2[i+k],pivot = 'middle', color='red', scale =1, units='xy')
            plt.scatter(XX2, YY2, c=WW2[i+k], vmin=-1.0, vmax=1.0, s=60, cmap=cm, edgecolors='none')
            plt.colorbar(ticks=[-1, 0, 1])
            plt.xlim([-1,N_x2])
            plt.ylim([-1,N_x2])
            ax = plt.subplot(3, r, 3+ i*r)	
            if i == 0:
                plt.title("fake", fontsize=24)
            #plt.quiver(XX3,YY3,UU3[i+k],VV3[i+k],pivot = 'middle', color='red', scale =1, units='xy')
            plt.scatter(XX3, YY3, c=WW3[i+k], vmin=-1.0, vmax=1.0, s=40, cmap=cm, edgecolors='none')
            plt.colorbar(ticks=[-1, 0, 1])
            plt.xlim([-1,N_x3])
            plt.ylim([-1,N_x3])	
               	
        plt.savefig(folder_name+"/"+labels[l]+".png")
        l = l+1
        plt.close()

    
def plot_stats(stat_list_real, stat_list_enc, stat_list_dec, folder_name):
    
    corr_real,dist = clip_correlation_range(stat_list_real[0], 10)
    corr_enc,dist = clip_correlation_range(stat_list_enc[0], 10)
    stat_list_dec[0],dist = clip_correlation_range(stat_list_dec[0], 10)
    #print(len(corr_real))
    #print(len(corr_real[0]))
    #print(len(stat_list_real[0]))
    #print(len(stat_list_real[0][0]))
    corr_list_real_submag = submag_from_correlation(corr_real, stat_list_real[3])
    corr_list_enc_submag = submag_from_correlation(corr_enc, stat_list_enc[3])
    corr_list_dec_submag = submag_from_correlation(stat_list_dec[0], stat_list_dec[3])
    
    #dist = dist_corr(20)
    temps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.04,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    temps = temps[:len(stat_list_real[1])]
    labels = ["T = 0.1", "T = 0.2", "T = 0.3","T = 0.4","T = 0.5","T = 0.6","T = 0.7","T = 0.8","T = 0.9","T = 1.0","T = 1.04","T = 1.1","T = 1.2","T = 1.3","T = 1.4","T = 1.5","T = 1.6","T = 1.7","T = 1.8","T = 1.9","T = 2.0"]
   
    l = 0
    for k in range(0,len(stat_list_real[1])):
        plt.figure(figsize=(35, 20))
        plt.suptitle(labels[l], fontsize=26)
        
        	
        ax = plt.subplot(3, 2, 1)		
        plt.title('correlation')
        list=zip(*sorted(zip(*(dist,corr_real[int(k)][0:]))))
        plt.plot(*list, label="N40 real")
           
        list=zip(*sorted(zip(*(dist,corr_enc[int(k)][0:]))))
        plt.plot(*list, label="renorm")
        list=zip(*sorted(zip(*(dist,stat_list_dec[0][int(k)][0:]))))
        plt.plot(*list, label="N40 fake")
        plt.ylim([-0.2,1.0])
        plt.legend()
        
        ax = plt.subplot(3, 2, 2)	
        plt.title('correlation mag subtracted')
        list=zip(*sorted(zip(*(dist,corr_list_real_submag[int(k)][0:]))))
        plt.plot(*list, label="N40 real")
        list=zip(*sorted(zip(*(dist,corr_list_enc_submag[int(k)][0:]))))
        plt.plot(*list, label="renorm")
        list=zip(*sorted(zip(*(dist,corr_list_dec_submag[int(k)][0:]))))
        plt.plot(*list, label="N40 fake")			
        #plt.xscale("log")
        #plt.yscale("log")
        plt.legend()
        
        
        ax = plt.subplot(3, 2, 3)
        plt.title('magnetisation')
        
        plt.plot(temps, stat_list_real[3], label="N40 real")
        plt.plot(temps, stat_list_enc[3], label="renorm")
        plt.plot(temps, stat_list_dec[3], label="N40 fake")   
        plt.legend()
         
         
        ax = plt.subplot(3, 2, 4)
        plt.title('suszeptibility')
         
        plt.plot(temps, stat_list_real[5], label="N40 real")
        plt.plot(temps, stat_list_enc[5], label="renorm")
        plt.plot(temps, stat_list_dec[5], label="N40 fake")   
        plt.legend()      
        
        ax = plt.subplot(3, 2, 5)
        plt.title('energy')
        
        plt.plot(temps, stat_list_real[1], label="N40 real")
        plt.plot(temps, stat_list_enc[1], label="renorm")
        plt.plot(temps, stat_list_dec[1], label="N40 fake")   
        plt.legend()
        
        ax = plt.subplot(3, 2, 6)
        plt.title('specific heat')
        
        plt.plot(temps, stat_list_real[6], label="N40 real")
        plt.plot(temps, stat_list_enc[6], label="renorm")
        plt.plot(temps, stat_list_dec[6], label="N40 fake")           
        plt.legend()
        	
        plt.savefig(folder_name+"/"+labels[l]+".png")
        l = l+1
        plt.close()

def make_sure_path_exists(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
        
def gen_and_plot_data(stat_samples, encoder, decoder, dis_noise):        
    #generate samples for plotting
    real_samples, enc_samples, dec_samples = get_samples_stats(stat_samples, encoder, decoder)
    #get stats of generated samples
    stat_list_real = get_stats(real_samples, BS)    
    stat_list_enc = get_stats(enc_samples, BS)        
    if n_feat_maps != 1:
        dec_img_rf, dec_img_batch, dec_img_all = seperate_decoded_samples(dec_samples)
        stat_list_dec_rf = get_stats(dec_img_rf, n_feat_maps)
        stat_list_dec_batch = get_stats(dec_img_batch, BS)
        stat_list_dec_all = get_stats(dec_img_all, BS*n_feat_maps)
        
        make_sure_path_exists(model_name+"/stat_rf_pic/")
        plot_stats(stat_list_real, stat_list_enc, stat_list_dec_rf, model_name+"/stat_rf_pic") 
        make_sure_path_exists(model_name+"/stat_batch_pic/")
        plot_stats(stat_list_real, stat_list_enc, stat_list_dec_batch, model_name+"/stat_batch_pic")     
        make_sure_path_exists(model_name+"/stat_all_pic/")
        plot_stats(stat_list_real, stat_list_enc, stat_list_dec_all, model_name+"/stat_all_pic")     

    else:
        dec_samples = [np.reshape(samp, (BS,3,40,40)) for samp in dec_samples]
        stat_list_dec = get_stats(dec_samples, BS) 
        make_sure_path_exists(model_name+"/stat_pic/")
        plot_stats(stat_list_real, stat_list_enc, stat_list_dec, model_name+"/stat_pic")  
        
    #generate samples for investigation the influence of noise
    if noise_dim:
        real_samples_noise, enc_samples_noise, dec_samples_noise, noise_samples, dec_noise_samples = get_samples_noise(stat_samples, encoder, decoder, dis_noise)
        
        if noise_dis:
            noise_samples_set = noise_samples[:]
            dec_noise_samples_set = dec_noise_samples[:]
            for i in range(1,n_batch_for_noise_plot):
                real_samples_noise, enc_samples_noise, dec_samples_noise, noise_samples, dec_noise_samples = get_samples_noise(stat_samples, encoder, decoder, dis_noise)            
                noise_samples_set = np.concatenate((noise_samples_set, noise_samples))
                dec_noise_samples_set = np.concatenate((dec_noise_samples_set, dec_noise_samples))

            make_sure_path_exists(model_name+"/noise_discrimation")
            
            with open(model_name+'/noise_discrimation/noise_samples.pickle', 'wb') as f:
                pickle.dump(noise_samples_set, f)        
            with open(model_name+'/noise_discrimation/dec_noise_samples.pickle', 'wb') as f:
                pickle.dump(dec_noise_samples_set, f)  
            plot_noise(noise_samples_set, dec_noise_samples_set, len(stat_samples),"noise_discrimation")
            
        
        
        #get stats of generated noise samples
        stat_list_real = get_stats(real_samples_noise, BS)    
        stat_list_enc = get_stats(enc_samples_noise, BS)        
        if n_feat_maps != 1:
            dec_img_rf_noise, dec_img_batch_noise, dec_img_all_noise = seperate_decoded_samples(dec_samples_noise)
            stat_list_dec_rf = get_stats(dec_img_rf_noise, n_feat_maps)
            stat_list_dec_batch = get_stats(dec_img_batch_noise, BS)
            stat_list_dec_all = get_stats(dec_img_all_noise, BS*n_feat_maps)
            
            make_sure_path_exists(model_name+"/stat_noise_rf_pic/")
            plot_stats(stat_list_real, stat_list_enc, stat_list_dec_rf, model_name+"/stat_noise_rf_pic")     
            make_sure_path_exists(model_name+"/stat_noise_batch_pic/")
            plot_stats(stat_list_real, stat_list_enc, stat_list_dec_batch, model_name+"/stat_noise_batch_pic")     
            make_sure_path_exists(model_name+"/stat_noise_all_pic/")
            plot_stats(stat_list_real, stat_list_enc, stat_list_dec_all, model_name+"/stat_noise_all_pic")   
            

  
        
    #plot samples        
    if n_feat_maps != 1:
        make_sure_path_exists(model_name+"/batch_pic/")
        plot_samples(BS,3,real_samples, enc_samples, dec_img_batch,model_name+"/batch_pic")
        real_samples, enc_samples= samples_for_randfeat_plot(real_samples, enc_samples)
        make_sure_path_exists(model_name+"/rf_pic/")
        plot_samples(n_feat_maps,3,real_samples, enc_samples, dec_img_rf, model_name+"/rf_pic")    
        if noise_dim:
            make_sure_path_exists(model_name+"/noise_batch_pic/")
            plot_samples(BS,3,real_samples_noise, enc_samples_noise, dec_img_batch_noise,model_name+"/noise_batch_pic")
            real_samples_noise, enc_samples_noise= samples_for_randfeat_plot(real_samples_noise, enc_samples_noise)
            make_sure_path_exists(model_name+"/noise_rf_pic/")
            plot_samples(n_feat_maps,3,real_samples_noise, enc_samples_noise, dec_img_rf_noise, model_name+"/noise_rf_pic")    

    else:
        make_sure_path_exists(model_name+"/pic/")
        plot_samples(BS,3,real_samples, enc_samples, dec_samples,model_name+"/pic")
        if noise_dim:
            make_sure_path_exists(model_name+"/noise_pic/")
            dec_samples_noise = [np.reshape(samp, (BS,3,40,40)) for samp in dec_samples_noise]
            plot_samples(BS,3,real_samples_noise, enc_samples_noise, dec_samples_noise,model_name+"/noise_pic")
   
        
        
        