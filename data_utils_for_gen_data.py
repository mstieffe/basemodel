import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from models_stat import *


K.set_image_dim_ordering('th')
noise_scale = 0.3
print("joooooooooO")
print(BS)
N_stats = BS * 100 * n_feat_maps
#jackknife stuff
bins = 10
bin_size = 300

############################ Load data and preprocessing #######################


def load_data_untrained():
    print("start loading data")
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.65.pickle', 'rb')
    x_train65 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.75.pickle', 'rb')
    x_train75 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.85.pickle', 'rb')
    x_train85 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.95.pickle', 'rb')
    x_train95 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.02.pickle', 'rb')
    x_train102 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.06.pickle', 'rb')
    x_train106 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.16.pickle', 'rb')
    x_train116 = pickle.load(file)
    file.close() 
    file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.26.pickle', 'rb')
    x_train126 = pickle.load(file)
    file.close()     
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
    
    print("loading data successfull")
    
    x_train = np.concatenate((x_train65, x_train75), axis = 0)
    x_train = np.concatenate((x_train, x_train85), axis = 0)
    x_train = np.concatenate((x_train, x_train95), axis = 0)
    x_train = np.concatenate((x_train, x_train102), axis = 0)
    x_train = np.concatenate((x_train, x_train106), axis = 0)
    x_train = np.concatenate((x_train, x_train116), axis = 0)
    x_train = np.concatenate((x_train, x_train126), axis = 0)
    x_train = np.concatenate((x_train, x_train17), axis = 0)
    x_train = np.concatenate((x_train, x_train18), axis = 0)
    x_train = np.concatenate((x_train, x_train19), axis = 0)
    x_train = np.concatenate((x_train, x_train20), axis = 0)


    stat_samples = [x_train65, x_train75, x_train85, x_train95, x_train102,x_train106,x_train116,x_train126, x_train17,x_train18,x_train19,x_train20]

    num_classes = len(stat_samples)+1

    y_train = np.array(5000*[0]+5000*[1]+5000*[2]+5000*[3]+5000*[4]+5000*[5]+5000*[6]+5000*[7]+5000*[8]+5000*[9]+5000*[10]+5000*[11])
    y_train = np_utils.to_categorical(y_train, num_classes = num_classes)    

    temps = [0.65, 0.75, 0.85, 0.95, 1.02, 1.06, 1.16, 1.26, 1.7, 1.8, 1.9, 2.0]

    return x_train, y_train, num_classes, stat_samples, temps



def ave_out_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],shape[2]])

    
def averager(layers):
    layers = tf.reduce_sum(layers, [1])
    layers = tf.div(layers, 500)
    return layers
    
def get_stats(data_set, temps):
    
    correlation_list = []
    correlation = []
    corr_err = []
    mag = []
    mag_err = []
    energy = []
    en_err = []
    sus = []
    sus_err = []
    spez = []
    spez_err =[]
    

    #print('Shape of input data: '+str(data_set[0].shape))
    t = 0
    for datas in data_set:
        corr_list = []
        mag_list = []
        en_list = []  
        en_sq_list = []
        mag_sq_list = [] 
        for b in range(0,bins):
            idx = range(b*bin_size,(b+1)*bin_size)
            data = datas[idx]
            (N,dim, N_x, N_y) = data.shape
            
            if N_x == 40:
                corr = get_corr.predict(data, batch_size=N)
                mp = get_mag.predict(data, batch_size=N)
                en = get_energy.predict(data, batch_size=N)
            else:
                corr = get_corr_enc.predict(data, batch_size=N)
                mp = get_mag_enc.predict(data, batch_size=N)
                en = get_energy_enc.predict(data, batch_size=N)
              
            #correlation  
            co_list = corr[:,:]
            
            corr = np.sum(corr, axis = 0)
            corr = corr / N		
            #corr = list(corr)    
    
    
            #magnetisation
            mp_sq = mp[:]
            mp_sq = np.square(mp_sq)   
            mp_sq = np.sum(mp_sq)
            mp_sq = mp_sq / N
          
            mp = np.sum(mp)
            mp = mp/ N

                        
            #energy
            en_sq = en[:]
            en_sq = np.square(en_sq)               
            en_sq = np.sum(en_sq)
            en_sq = en_sq / N
    
            
            en = np.sum(en)
            en= en / N

            mp2 = mp*mp
            corr = corr - np.full(100, mp2)
            corr = corr / (1-mp2)
 
            co_list = co_list - np.full((N,100), mp2)
            co_list = co_list / (1-mp2)
            
            if b == 0:
                correlation_list.append(co_list)
          
            corr_list.append(corr)
            mag_list.append(mp)
            en_list.append(en)   
            en_sq_list.append(en_sq)  
            mag_sq_list.append(mp_sq)      
        
        sus_list = []
        for n in range(0,len(mag_list)):
            sus_list.append((mag_sq_list[n] - mag_list[n]*mag_list[n])/temps[t])
                      
        spez_list = []
        for n in range(0,len(en_list)):
            spez_list.append((en_sq_list[n] - en_list[n]*en_list[n])/(temps[t]*temps[t]))           
           
           
        corr_jk = []
        for n in range(0,bins):
            jk = corr_list[:n] + corr_list[n+1:]
            jk = sum(jk)/(bins-1)
            corr_jk.append(jk)
            
        sus_jk = []
        for n in range(0,bins):
            jk = sus_list[:n] + sus_list[n+1:]
            jk = sum(jk)/(bins-1)
            sus_jk.append(jk)

        spez_jk = []
        for n in range(0,bins):
            jk = spez_list[:n] + spez_list[n+1:]
            jk = sum(jk)/(bins-1)
            spez_jk.append(jk)           

        mag_jk = []
        for n in range(0,bins):
            jk = mag_list[:n] + mag_list[n+1:]
            jk = sum(jk)/(bins-1)
            mag_jk.append(jk)   
 
        en_jk = []
        for n in range(0,bins):
            jk = en_list[:n] + en_list[n+1:]
            jk = sum(jk)/(bins-1)
            en_jk.append(jk)   

         
        t = t+1    
            
        corr_mean = sum(corr_jk)/bins
        corr_jk = [(s-corr_mean)*(s-corr_mean) for s in corr_jk]
        corr_jk = np.sqrt(sum(corr_jk)*(bins-1)/bins)        
        correlation.append(list(corr_mean))
        corr_err.append(list(corr_jk))
            
        sus_mean = sum(sus_jk)/bins
        sus_jk = [(s-sus_mean)*(s-sus_mean) for s in sus_jk]
        sus_jk = np.sqrt(sum(sus_jk)*(bins-1)/bins)        
        sus.append(sus_mean)
        sus_err.append(sus_jk)
        
        spez_mean = sum(spez_jk)/bins
        spez_jk = [(s-spez_mean)*(s-spez_mean) for s in spez_jk]
        spez_jk = np.sqrt(sum(spez_jk)*(bins-1)/bins)          
        spez.append(spez_mean)
        spez_err.append(spez_jk)
        
        mag_mean = sum(mag_jk)/bins
        mag_jk = [(s-mag_mean)*(s-mag_mean) for s in mag_jk]
        mag_jk = np.sqrt(sum(mag_jk)*(bins-1)/bins)          
        mag.append(mag_mean)
        mag_err.append(mag_jk)

        en_mean = sum(en_jk)/bins
        en_jk = [(s-en_mean)*(s-en_mean) for s in en_jk]
        en_jk = np.sqrt(sum(en_jk)*(bins-1)/bins)          
        energy.append(en_mean)
        en_err.append(en_jk)        
        

    return [correlation, corr_err, energy, en_err , mag ,mag_err, sus, sus_err, spez, spez_err, correlation_list]

def samples_for_randfeat_plot(real_samples, enc_samples):
    real_img_rf = []
    for samples in real_samples:
        real_img_rf.append(np.repeat(samples[[0],:,:,:], n_feat_maps, axis = 0))
    enc_img_rf = []
    for samples in enc_samples:
        enc_img_rf.append(np.repeat(samples[[0],:,:,:], n_feat_maps, axis = 0))    
    return real_img_rf,enc_img_rf

def get_samples_noise(stat_samples, encoder, decoder, dis_noise, size):
    idx = np.random.choice(np.arange(stat_samples[0].shape[0]), size, replace=False)
    samples = stat_samples[0][idx]
    samples = np.repeat(samples[[0],:,:,:], size, axis = 0)
    real_samples = [samples]
    enc_samples = [encoder.predict(real_samples, batch_size=BS)]

    noise_sample = make_noise_samples(noise_scale, size)
    noise_samples = [noise_sample]
    dec_samples = [decoder.predict([enc_samples[-1], noise_sample], batch_size=BS) ] 
    if noise_dis:
        dec_noise_samples =  [dis_noise.predict(dec_samples[-1][:,0,:,:,:])]       
    else:
        dec_noise_samples = []
        
    
    for n in range(1, len(stat_samples)):
        idx = np.random.choice(np.arange(stat_samples[n].shape[0]), size, replace=False)
        samples = stat_samples[n][idx]
        samples = np.repeat(samples[[0],:,:,:], size, axis = 0)
        real_samples.append(samples)
        enc_samples.append(encoder.predict(real_samples[-1]))

        noise_sample = make_noise_samples(noise_scale, size)
        noise_samples.append(noise_sample)
        dec_samples.append(decoder.predict([enc_samples[-1], noise_sample] , batch_size=BS))  
        if noise_dis:
            dec_noise_samples.append(dis_noise.predict(dec_samples[-1][:,0,:,:,:]))

    return real_samples, enc_samples, dec_samples, noise_samples, dec_noise_samples

def get_samples_noise_batch(stat_samples, encoder, decoder, size, bs_number):
    
    for n in range(0,bs_number):
        idx = np.random.choice(np.arange(stat_samples[0].shape[0]), size, replace=False)
        samples = stat_samples[0][idx]
        samples = np.repeat(samples[[0],:,:,:], size, axis = 0)
        enc_samp = encoder.predict(samples, batch_size=BS)
    
        noise_sample = make_noise_samples(noise_scale, size)
        if n == 0:
            enc_sampl = enc_samp
            dec_samp = decoder.predict([enc_samp, noise_sample], batch_size=BS)
            real_samp = samples
        else:
            enc_sampl = np.concatenate((enc_sampl, enc_samp), axis = 0)
            dec_samp = np.concatenate((dec_samp, decoder.predict([enc_samp, noise_sample], batch_size=BS)), axis = 0)
            real_samp = np.concatenate((real_samp, samples), axis = 0)
    enc_samples = [enc_sampl]
    real_samples = [real_samp]
    dec_samples = [dec_samp ] 

        
    
    for k in range(1, len(stat_samples)):
        for n in range(0,bs_number):
            idx = np.random.choice(np.arange(stat_samples[k].shape[0]), size, replace=False)
            samples = stat_samples[k][idx]
            samples = np.repeat(samples[[0],:,:,:], size, axis = 0)
            enc_samp = encoder.predict(samples, batch_size=BS)
        
            noise_sample = make_noise_samples(noise_scale, size)
            if n == 0:
                enc_sampl = enc_samp
                dec_samp = decoder.predict([enc_samp, noise_sample], batch_size=BS)
                real_samp = samples
            else:
                enc_sampl = np.concatenate((enc_sampl, enc_samp), axis = 0)
                dec_samp = np.concatenate((dec_samp, decoder.predict([enc_samp, noise_sample], batch_size=BS)), axis = 0)
                real_samp = np.concatenate((real_samp, samples), axis = 0)
        enc_samples.append(enc_sampl)
        real_samples.append(real_samp)
        dec_samples.append(dec_samp)


    return real_samples, enc_samples, dec_samples



def get_samples_noise_for_pearson(stat_samples, encoder, decoder, size, bs_number):
    
    for n in range(0,bs_number):
        idx = np.random.choice(np.arange(stat_samples[0].shape[0]), size, replace=False)
        samples = stat_samples[0][idx]
        samples = np.repeat(samples[[0],:,:,:], size, axis = 0)
        enc_samp = encoder.predict(samples, batch_size=BS)
    
        noise_sample = make_noise_samples(noise_scale, size)
        if n == 0:
            enc_sampl = enc_samp
            dec_samp = np.reshape(decoder.predict([enc_samp, noise_sample], batch_size=BS), (1,32,3,40,40))
            real_samp = samples
        else:
            enc_sampl = np.concatenate((enc_sampl, enc_samp), axis = 0)
            dec_samp = np.concatenate((dec_samp, np.reshape(decoder.predict([enc_samp, noise_sample], batch_size=BS), (1,32,3,40,40))), axis = 0)
            real_samp = np.concatenate((real_samp, samples), axis = 0)
    enc_samples = [enc_sampl]
    real_samples = [real_samp]
    dec_samples = [dec_samp ] 

        
    
    for k in range(1, len(stat_samples)):
        for n in range(0,bs_number):
            idx = np.random.choice(np.arange(stat_samples[k].shape[0]), size, replace=False)
            samples = stat_samples[k][idx]
            samples = np.repeat(samples[[0],:,:,:], size, axis = 0)
            enc_samp = encoder.predict(samples, batch_size=BS)
        
            noise_sample = make_noise_samples(noise_scale, size)
            if n == 0:
                enc_sampl = enc_samp
                dec_samp = np.reshape(decoder.predict([enc_samp, noise_sample], batch_size=BS), (1,32,3,40,40))
                real_samp = samples
            else:
                enc_sampl = np.concatenate((enc_sampl, enc_samp), axis = 0)
                dec_samp = np.concatenate((dec_samp, np.reshape(decoder.predict([enc_samp, noise_sample], batch_size=BS), (1,32,3,40,40))), axis = 0)
                real_samp = np.concatenate((real_samp, samples), axis = 0)
        enc_samples.append(enc_sampl)
        real_samples.append(real_samp)
        dec_samples.append(dec_samp)


    return real_samples, enc_samples, dec_samples


def get_samples_stats(stat_samples, encoder, decoder, size): 
    #print("generate data")
    idx = np.random.choice(np.arange(stat_samples[0].shape[0]), size, replace=False)
    real_samples = [stat_samples[0][idx]]
    enc_samples = [encoder.predict(real_samples)]

    if noise_dim != 0:
        noise_sample = make_noise_samples(noise_scale, N_stats)
        dec_samples = [decoder.predict([enc_samples[-1], noise_sample] , batch_size=BS) ]        
    else:
        dec_samples = [decoder.predict([enc_samples[-1]] , batch_size=BS*n_feat_maps)]     
        #generation of decoded samples with random layer
        """
        dec_samp = decoder.predict(enc_samples[-1][0:(0+1)*BS])
        for n in range(1, int(size/BS)):
            #print(dec_samp.shape)
            dec_samp = np.concatenate((dec_samp,decoder.predict(enc_samples[-1][n*BS:(n+1)*BS])), axis = 0)
        dec_samples = [dec_samp]
        """
    for n in range(1, len(stat_samples)):
        #print("temperature: "+str(n) )
        idx = np.random.choice(np.arange(stat_samples[n].shape[0]), size, replace=False)
        real_samples.append(stat_samples[n][idx])

        enc_samples.append(encoder.predict(real_samples[-1]))
        if noise_dim != 0:
            noise_sample = make_noise_samples(noise_scale, N_stats)
            dec_samples.append(decoder.predict([enc_samples[-1], noise_sample], batch_size=BS))  
        else:
            dec_samples.append(decoder.predict([enc_samples[-1]], batch_size=BS))
            """
            dec_samp = decoder.predict(enc_samples[-1][0:(0+1)*BS])
            for n in range(1, int(size/BS)):
                dec_samp = np.concatenate((dec_samp,decoder.predict(enc_samples[-1][n*BS:(n+1)*BS])), axis = 0)
            dec_samples.append(dec_samp)
            """
    return real_samples, enc_samples, dec_samples
    
def seperate_decoded_samples(dec_samples, size):
    dec_img_rf = []
    dec_img_batch = []
    dec_img_all = []
    for samples in dec_samples:
        dec_img_rf.append(samples[0,:,:,:,:])
        dec_img_batch.append(samples[:,0,:,:,:])
        dec_all = np.reshape(samples, (size*n_feat_maps,3,40,40))
        #for j in range(1,BS):
        #    dec_all = np.concatenate((dec_all, samples[j,:,:,:,:]), axis = 0)
        dec_img_all.append(dec_all)
    #print(dec_img_rf[0].shape)
    #print(dec_img_batch[0].shape)
    #print(dec_img_all[0].shape)

    return dec_img_rf, dec_img_batch, dec_img_all


def clip_correlation_range(correlation_list,corr_error, max_dis= 10):
    #print(len(correlation_list))
    #print(len(correlation_list[0]))
    corr_list = []
    corr_err = []
    d = []
    for n in range(0,len(correlation_list)):
        corr = []
        d = []
        err = []
        dist = dist_corr(10)
        for i in range(0,len(dist)):
            if dist[i] >= 0.0  and dist[i]<=max_dis:
                d.append(dist[i])
                corr.append(correlation_list[n][i])    
                err.append(corr_error[n][i])
        corr_list.append(corr)
        corr_err.append(err)
    return corr_list,corr_err, d

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
    
def make_noise_samples(scale, size):
    noise_sample = scale * np.random.randn(size, noise_dim)
    return noise_sample
    
    
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
            plt.xlim([0,noise_dim])
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
    
    print(len(stat_list_real[0]))
    corr_real, corr_err_real, dist = clip_correlation_range(stat_list_real[0],stat_list_real[1], 10)
    corr_enc, corr_err_enc, dist = clip_correlation_range(stat_list_enc[0],stat_list_enc[1], 10)
    corr_dec, corr_err_dec,dist = clip_correlation_range(stat_list_dec[0],stat_list_dec[1], 10)
    #print(len(corr_real))
    #print(len(corr_real[0]))
    #print(len(stat_list_real[0]))
    #print(len(stat_list_real[0][0]))
    #corr_list_real_submag = submag_from_correlation(corr_real, stat_list_real[3])
    #corr_list_enc_submag = submag_from_correlation(corr_enc, stat_list_enc[3])
    #corr_list_dec_submag = submag_from_correlation(stat_list_dec[0], stat_list_dec[3])
    
    #dist = dist_corr(20)
    temps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.04,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    temps = temps[:len(stat_list_real[1])]
    labels = ["T = 0.1", "T = 0.2", "T = 0.3","T = 0.4","T = 0.5","T = 0.6","T = 0.7","T = 0.8","T = 0.9","T = 1.0","T = 1.04","T = 1.1","T = 1.2","T = 1.3","T = 1.4","T = 1.5","T = 1.6","T = 1.7","T = 1.8","T = 1.9","T = 2.0"]
   
    l = 0
    for k in range(0,len(stat_list_real[1])):
        font = {     'size'   : 20}
        
        plt.rc('font', **font)
        


        list2=zip(*sorted(zip(*(dist,corr_err_real[k][0:]))))
        d1, var1 = list2
        list1=zip(*sorted(zip(*(dist,corr_real[k]))))
        d, corr = list1  
        plt.plot(d, corr, label="Original", color = 'darkblue')  
        plt.errorbar(d, corr, yerr=var1, fmt='.', color = 'darkblue', alpha = 1.0)
        
        list2=zip(*sorted(zip(*(dist,corr_err_enc[k][0:]))))
        d1, var1 = list2
        list1=zip(*sorted(zip(*(dist,corr_enc[k]))))
        d, corr = list1  
        plt.plot(d, corr, label="Renormalized", color = 'darkgreen')
        plt.errorbar(d, corr, yerr=var1, fmt='.', color = 'darkgreen', alpha = 1.0)

        list2=zip(*sorted(zip(*(dist,corr_err_dec[k][0:]))))
        d1, var1 = list2
        list1=zip(*sorted(zip(*(dist,corr_dec[k]))))
        d, corr = list1  
        plt.plot(d, corr, label="Network", color = 'darkred')
        plt.errorbar(d, corr, yerr=var1, fmt='.', color = 'darkred', alpha = 1.0)
        
        plt.xlabel("Distance "+r'$r$', fontweight='bold')
        plt.ylabel("Correlation "+r'$\tilde{C}(r)$', labelpad=0, fontweight='bold')
        plt.legend(fontsize= 18)
        
        plt.xlim(1.0,10.0)
        plt.ylim(-0.1,0.4)
        
        plt.savefig(folder_name+"/"+labels[l]+".png", bbox_inches='tight')
        #plt.savefig("corr_renorm.pdf", bbox_inches='tight')
        plt.close()
        print("done")
        
        """
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
        """


def pearson_corr_of_set(samples, number):
    (N,dim,N_x,N_y) = samples.shape
    pearson = 0
    for n in range(0,number):
        rid1 = np.random.randint(N)
        rid2 = np.random.randint(N)
        while rid1 == rid2:
            rid2 = np.random.randint(N)
        a = samples[rid1]
        b = samples[rid2]
        r = pearsonr(a.flatten(),b.flatten())
        pearson = r[0] +pearson 
    pearson = pearson / number
    return pearson
    
def pearson_corr_of_set2(samples, number):
    (N,dim,N_x,N_y) = samples.shape
    pearson = np.zeros((dim,N_x,N_y))
    mean = np.sum(samples, axis = 0) / N
    var = np.zeros((dim,N_x,N_y))
    for samp in samples:
        var = np.multiply(samp-mean,samp-mean) + var
    var = var / (N-1)
    for n in range(0,N):
        for m in range(0,N):
            p = np.multiply((samples[n,:,:,:]-mean),(samples[m,:,:,:]-mean) )
            p = np.divide(p,var)
            pearson = pearson + p
    pearson = pearson /(N*N)
    pearson = np.sum(pearson)/(dim*N_x*N_y)
    return pearson
    
def get_pearson_corr(samples, size, number):
    pearson_corr_x = []
    pearson_corr_z = []
    pearson_corr_y = []
    pearson = []
    t = 0
    for sample in samples:
        #print(t)
        p_x = []
        p_y = []
        p_z = []
        p = []
        for n in range(0,int(size/BS)):
            #print(sample.shape)
            samp = sample[n]
            (N,dim,N_x,N_y) = samp.shape
            if N_x == 40:
                p.append(get_p.predict(np.reshape(samp, (1,BS,3,40,40))))
            elif N_x == 20:
                p.append(get_pear_N20.predict(np.reshape(samp, (1,BS,3,20,20))))
            elif N_x == 10:
                p.append(get_pear_N10.predict(np.reshape(samp, (1,BS,3,10,10))))
            #print(samp.shape)
            #print(s.shape)
            p_x.append(pearson_corr_of_set(samp[:,[0],:,:], number) )
            p_y.append(pearson_corr_of_set(samp[:,[1],:,:], number) )
            p_z.append(pearson_corr_of_set(samp[:,[2],:,:], number) )
        pearson.append(p)
        pearson_corr_x.append(p_x)
        pearson_corr_y.append(p_y)
        pearson_corr_z.append(p_z)
        t = t+1
    return pearson_corr_x,pearson_corr_y,pearson_corr_z, pearson
        
def get_pearson_corr_rf(samples, size, number):
    pearson_corr_x = []
    pearson_corr_z = []
    pearson_corr_y = []
    pearson = []
    t = 0
    for sample in samples:
        #print(t)
        p_x = []
        p_y = []
        p_z = []
        p = []
        (NN,N,dim,N_x,N_y) = sample.shape
        for n in range(0,number):
            #print(sample.shape)
            samp = sample[n]
            (N,dim,N_x,N_y) = samp.shape
            if N_x == 40:
                p.append(get_p.predict(np.reshape(samp, (1,n_feat_maps,3,40,40))))
            elif N_x == 20:
                p.append(get_pear_N20.predict(np.reshape(samp, (1,n_feat_maps,3,20,20))))
            elif N_x == 10:
                p.append(get_pear_N10.predict(np.reshape(samp, (1,n_feat_maps,3,10,10))))
            #print(samp.shape)
            #print(s.shape)
            p_x.append(pearson_corr_of_set(samp[:,[0],:,:], number) )
            p_y.append(pearson_corr_of_set(samp[:,[1],:,:], number) )
            p_z.append(pearson_corr_of_set(samp[:,[2],:,:], number) )
        pearson.append(p)
        pearson_corr_x.append(p_x)
        pearson_corr_y.append(p_y)
        pearson_corr_z.append(p_z)
        t = t+1
    return pearson_corr_x,pearson_corr_y,pearson_corr_z, pearson      
    

def make_sure_path_exists(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
        
def gen_and_save_data(stat_samples, encoder, decoder, dis_noise, folder_name, temps):    
    #generate samples for plotting
    real_samples, enc_samples, dec_samples = get_samples_stats(stat_samples, encoder, decoder, N_stats)
    #get stats of generated samples
    stat_list_real = get_stats(real_samples, temps)    
    stat_list_enc = get_stats(enc_samples, temps)     
        
    print("got samples")
    
    if n_feat_maps != 1:
        make_sure_path_exists(folder_name+"/stat_rf/")
        pearson_corr_x,pearson_corr_y,pearson_corr_z, pearson = get_pearson_corr_rf(dec_samples, N_stats, 100)
        with open(folder_name+'/stat_rf/pearson.pickle', 'wb') as f:
            pickle.dump(pearson, f)  
        with open(folder_name+'/stat_rf/pearson_corr_x.pickle', 'wb') as f:
            pickle.dump(pearson_corr_x, f)  
        with open(folder_name+'/stat_rf/pearson_corr_y.pickle', 'wb') as f:
            pickle.dump(pearson_corr_y, f)  
        with open(folder_name+'/stat_rf/pearson_corr_z.pickle', 'wb') as f:
            pickle.dump(pearson_corr_z, f)  
        dec_img_rf, dec_img_batch, dec_img_all = seperate_decoded_samples(dec_samples, N_stats)
        #stat_list_dec_rf = get_stats(dec_img_rf, n_feat_maps)
        stat_list_dec_batch = get_stats(dec_img_batch, temps)
        stat_list_dec_all = get_stats(dec_img_all, temps)
        
        with open(folder_name+'/stat_rf/stat_list_real.pickle', 'wb') as f:
            pickle.dump(stat_list_real, f)  
        with open(folder_name+'/stat_rf/stat_list_enc.pickle', 'wb') as f:
            pickle.dump(stat_list_enc, f)             
        #with open(folder_name+'/stat_rf/stat_list_dec_rf.pickle', 'wb') as f:
        #    pickle.dump(stat_list_dec_rf, f)  
        with open(folder_name+'/stat_rf/stat_list_dec_batch.pickle', 'wb') as f:
            pickle.dump(stat_list_dec_batch, f)      
        with open(folder_name+'/stat_rf/stat_list_dec_all.pickle', 'wb') as f:
            pickle.dump(stat_list_dec_all, f)                

    else:
        dec_samples = [np.reshape(samp, (N_stats,3,40,40)) for samp in dec_samples]
        stat_list_dec = get_stats(dec_samples, temps) 
        make_sure_path_exists(folder_name+"/stat/")
        with open(folder_name+'/stat/stat_list_real.pickle', 'wb') as f:
            pickle.dump(stat_list_real, f)  
        with open(folder_name+'/stat/stat_list_enc.pickle', 'wb') as f:
            pickle.dump(stat_list_enc, f)       
        with open(folder_name+'/stat/stat_list_dec.pickle', 'wb') as f:        
            pickle.dump(stat_list_dec, f)  
    print("got stats")   
    #generate samples for investigation the influence of noise
    if noise_dim:
        real_samples_noise, enc_samples_noise, dec_samples_noise, noise_samples, dec_noise_samples = get_samples_noise(stat_samples, encoder, decoder, dis_noise, N_stats)
        
        if noise_dis:
            noise_samples_set = noise_samples[:]
            dec_noise_samples_set = dec_noise_samples[:]
            for i in range(1,2):
                real_samples_noise, enc_samples_noise, dec_samples_noise, noise_samples, dec_noise_samples = get_samples_noise(stat_samples, encoder, decoder, dis_noise,N_stats)            
                noise_samples_set = np.concatenate((noise_samples_set, noise_samples))
                dec_noise_samples_set = np.concatenate((dec_noise_samples_set, dec_noise_samples))

            make_sure_path_exists(folder_name+"/noise_discrimation")
            
            with open(folder_name+'/noise_discrimation/noise_samples.pickle', 'wb') as f:
                pickle.dump(noise_samples_set, f)        
            with open(folder_name+'/noise_discrimation/dec_noise_samples.pickle', 'wb') as f:
                pickle.dump(dec_noise_samples_set, f)  
            #plot_noise(noise_samples_set, dec_noise_samples_set, len(stat_samples),"noise_discrimation")
            
        
        
        #get stats of generated noise samples
        stat_list_real = get_stats(real_samples_noise, temps)    
        stat_list_enc = get_stats(enc_samples_noise, temps)        
        if n_feat_maps != 1:
            dec_img_rf_noise, dec_img_batch_noise, dec_img_all_noise = seperate_decoded_samples(dec_samples_noise, N_stats)
            #stat_list_dec_rf = get_stats(dec_img_rf_noise, n_feat_maps)
            stat_list_dec_batch = get_stats(dec_img_batch_noise, temps)
            stat_list_dec_all = get_stats(dec_img_all_noise, temps)

            make_sure_path_exists(folder_name+"/stat_noise/")
            with open(folder_name+'/stat_noise/stat_list_real.pickle', 'wb') as f:
                pickle.dump(stat_list_real, f)  
            with open(folder_name+'/stat_noise/stat_list_enc.pickle', 'wb') as f:
                pickle.dump(stat_list_enc, f)             
            #with open(folder_name+'/stat_noise/stat_list_dec_rf.pickle', 'wb') as f:
            #    pickle.dump(stat_list_dec_rf, f)  
            with open(folder_name+'/stat_noise/stat_list_dec_batch.pickle', 'wb') as f:
                pickle.dump(stat_list_dec_batch, f)      
            with open(folder_name+'/stat_noise/stat_list_dec_all.pickle', 'wb') as f:
                pickle.dump(stat_list_dec_all, f)   
  
    real_samples, enc_samples, dec_samples = get_samples_stats(stat_samples, encoder, decoder, BS)
    if n_feat_maps != 1:
        dec_img_rf, dec_img_batch, dec_img_all = seperate_decoded_samples(dec_samples, BS)  
    else:
        dec_samples = [np.reshape(samp, (BS,3,40,40)) for samp in dec_samples]        
    #plot samples        
        
    if n_feat_maps != 1:
        
        make_sure_path_exists(folder_name+"/samples/")
        with open(folder_name+'/samples/real_samples.pickle', 'wb') as f:
            pickle.dump(real_samples, f)  
        with open(folder_name+'/samples/enc_samples.pickle', 'wb') as f:
            pickle.dump(enc_samples, f)             
        with open(folder_name+'/samples/dec_img_batch.pickle', 'wb') as f:
            pickle.dump(dec_img_batch, f)  
        with open(folder_name+'/samples/dec_img_rf.pickle', 'wb') as f:
            pickle.dump(dec_img_rf, f)      
        

        if noise_dim:
            real_samples_noise, enc_samples_noise, dec_samples_noise = get_samples_noise_batch(stat_samples, encoder, decoder, BS, 1)
            dec_img_rf_noise, dec_img_batch_noise, dec_img_all_noise = seperate_decoded_samples(dec_samples_noise, BS)           
            make_sure_path_exists(folder_name+"/samples_noise/")
            with open(folder_name+'/samples_noise/real_samples_noise.pickle', 'wb') as f:
                pickle.dump(real_samples_noise, f)  
            with open(folder_name+'/samples_noise/enc_samples_noise.pickle', 'wb') as f:
                pickle.dump(enc_samples_noise, f)             
            with open(folder_name+'/samples_noise/dec_img_batch_noise.pickle', 'wb') as f:
                pickle.dump(dec_img_batch_noise, f)  
            with open(folder_name+'/samples_noise/dec_img_rf_noise.pickle', 'wb') as f:
                pickle.dump(dec_img_rf_noise, f)              
    else:
        make_sure_path_exists(folder_name+"/samples/")

        with open(folder_name+'/samples/real_samples.pickle', 'wb') as f:
            pickle.dump(real_samples, f)  
        with open(folder_name+'/samples/enc_samples.pickle', 'wb') as f:
            pickle.dump(enc_samples, f)             
        with open(folder_name+'/samples/dec_samples.pickle', 'wb') as f:
            pickle.dump(dec_samples, f)         
        print("get pearson")
        if noise_dim:
            make_sure_path_exists(folder_name+"/samples_noise/")
            real_samples_noise, enc_samples_noise, dec_samples_noise = get_samples_noise_batch(stat_samples, encoder, decoder, BS, 1)
    
            with open(folder_name+'/samples_noise/real_samples_noise.pickle', 'wb') as f:
                pickle.dump(real_samples_noise, f)  
            with open(folder_name+'/samples_noise/enc_samples_noise.pickle', 'wb') as f:
                pickle.dump(enc_samples_noise, f)             
            with open(folder_name+'/samples_noise/dec_samples_noise.pickle', 'wb') as f:
                pickle.dump(dec_samples_noise, f)
                
            real_samples_noise, enc_samples_noise, dec_samples_noise = get_samples_noise_for_pearson(stat_samples, encoder, decoder, BS, 100)
            pearson_corr_x,pearson_corr_y,pearson_corr_z, pearson = get_pearson_corr(dec_samples_noise, 100*BS, 100)
            with open(folder_name+'/samples_noise/pearson_corr_x.pickle', 'wb') as f:
                pickle.dump(pearson_corr_x, f)  
            with open(folder_name+'/samples_noise/pearson_corr_y.pickle', 'wb') as f:
                pickle.dump(pearson_corr_y, f)  
            with open(folder_name+'/samples_noise/pearson_corr_z.pickle', 'wb') as f:
                pickle.dump(pearson_corr_z, f)  
            with open(folder_name+'/samples_noise/pearson.pickle', 'wb') as f:
                pickle.dump(pearson, f)     
        
        
        