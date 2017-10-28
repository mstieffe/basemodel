import os

from config import *
from models import *
from models_stat import *
from data_utils import *
from data_utils_for_gen_data import *
from training import *
from shutil import copyfile


#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')


################## load data ####################################
X_train, Y_train, num_classes, stat_samples, temps = load_data()
X_train_u, Y_train_u, num_classes_u, stat_samples_u, temps_u = load_data_untrained()

################## get stats of data ############################
#stat_list = get_stats(stat_samples, 1000)

################## Compile models ###############################
encoder = make_encoder()
encoder.compile(loss='categorical_crossentropy', optimizer=gan_opt)

if noise_dim == 0:
    if dec_dense:
        decoder = make_decoder_dense(model_name)
        decoder.compile(loss='categorical_crossentropy', optimizer=gan_opt)    
    else:
        decoder = make_decoder(model_name)
        decoder.compile(loss='categorical_crossentropy', optimizer=gan_opt)
else:
    dec_noise, dec_noise2, decoder = make_decoder_noise()
    dec_noise.compile(loss='categorical_crossentropy', optimizer=gan_opt)
    dec_noise2.compile(loss='categorical_crossentropy', optimizer=gan_opt)
    make_trainable(dec_noise,True)
    make_trainable(dec_noise2,True)
    decoder.compile(loss='categorical_crossentropy', optimizer=gan_opt)

#dec_sing = decoder
dec_sing = make_dec_sing(decoder)
dec_sing.compile(loss='categorical_crossentropy', optimizer=gan_opt)

if noise_dis:
    dis_temp, dis_noise, dis = make_dis_temp_and_noise(num_classes)
else:
    dis_temp = make_dis_temp(num_classes)
    dis_temp.compile(loss='categorical_crossentropy', optimizer=dis_opt, metrics=['accuracy'])
    dis_noise = 0
    dis = 0


make_trainable(dis_temp,False)
if noise_dim == 0:
    make_trainable(decoder,True)
    gan = make_gan(decoder, dis_temp, encoder, dis_noise)
else:
    make_trainable(dec_noise,True)
    make_trainable(dec_noise2,True)
    gan = make_gan_noise(dec_noise, dec_noise2, encoder, dis_temp, dis_noise)    

gan_loss = ['categorical_crossentropy'] #loss for adversarial
if recon:
    gan_loss.append('mse') #reconstruction loss
if pearson_loss:
    gan_loss.append('mae') #reconstruction loss
if noise_dis:
    gan_loss.append('mse') #noise discrimination loss


lr_ad = K.variable(loss_weight_ad)
l_weights = [lr_ad]
if recon:
    lr_rec = K.variable(loss_weight_recon)
    l_weights.append(lr_rec)
if noise_dis:
    lr_noise_dis = K.variable(loss_weight_noise_dis)
    l_weights.append(lr_noise_dis)
if pearson_loss:
    lr_pearson = K.variable(loss_weight_pearson)
    l_weights.append(lr_pearson)

gan.compile(loss=gan_loss, optimizer=gan_opt, metrics=['accuracy'], loss_weights=l_weights)


################## training #####################################
pre_training(X_train, Y_train,encoder, dec_sing, dis_temp,num_classes, 100)

if noise_dis:
    losses = {"g":[], "dt":[], "dn":[]} 
else:
    losses = {"g":[], "dt":[]} 
    
for n in range(0,nb_epoch):

    losses = training(losses,X_train, Y_train,stat_samples,encoder, dec_sing, dis_temp,dis_noise, dis,gan,num_classes,nb_batches_per_epoch)
    #K.set_value(alpha, K.get_value(alpha) * 10)
    
    """
    print("new learning rates:")
    for l in l_weights:
        print(K.get_value(l))
    
    lr_ad = increase_lr(n, n_max, lr_ad, inc_ad)
    if recon:
        lr_rec = increase_lr(n, n_max, lr_rec, inc_rec)
    if pearson_loss and n > 2:
        lr_pearson = increase_lr(n, n_max, lr_pearson, inc_pearson)
    if noise_dis:
        lr_noise_dis = increase_lr(n, n_max, lr_noise_dis, inc_noise)
    """
    
    if n == 0:
        make_sure_path_exists(model_name+"/")
        make_sure_path_exists(model_name+"/model")
        make_sure_path_exists(model_name+"/model/data")    
        make_sure_path_exists(model_name+"/model/data_u")    
        copyfile("config.py", model_name+"/model/config.py")
    
    if save_model:
        print("save_model")
        with open(model_name+'/model/loss.pickle', 'wb') as f:
            pickle.dump(losses, f)
        
        decoder.save(model_name+'/model/decoder.h5', overwrite=True)
        dis_temp.save(model_name+'/model/dis_temp.h5', overwrite=True)
        if noise_dis:
            dis_noise.save(model_name+'/model/dis_noise.h5', overwrite=True)  
        decoder.save_weights(model_name+'/model/decoder_weights.h5', overwrite=True)		
        dis_temp.save_weights(model_name+'/model/dis_temp_weights.h5', overwrite=True)	
        if noise_dis:
            dis_noise.save_weights(model_name+'/model/dis_noise_weights.h5', overwrite=True)	
    
    if plot_data:
        print("generate and plot data")
        gen_and_save_data(stat_samples, encoder, decoder, dis_noise, model_name+"/data", temps)
        gen_and_save_data(stat_samples_u, encoder, decoder, dis_noise, model_name+"/data_u", temps_u)













