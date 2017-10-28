from keras.optimizers import *

########### parameters ##############
model_name = "name_of_model 

#keras 2.0, pearson = sum p_i (not squared), vectorized pearson (not only z component)

#network architecture
n_feat_maps = 32 #number sets of feature maps for feature map dropout
noise_dim = 0 #dimension for the additional noise input, if no noise set noise_dim = 0
noise_scale = 0.3
nb_ch_dec = 100 #number of channels for the decoder to start with
nb_ch_dis = 256 #number of channels for the discriminator to start with
dr_dis = 0.3 #droprate for discriminator

#loss specifications
recon = 1 #reconstruction error
noise_dis = 0 #discrimator for the noise input (like in stacked gan paper), yes = 1, no = 0
pearson_loss = 1 #Pearson Loss, average of square of all sample pairs

#Fully convolutional or dense model
dec_dense = 0

#Loss weights
loss_weight_ad = 1.0
loss_weight_recon = 10.0
loss_weight_pearson = 0.0
loss_weight_noise_dis = 0.0

#Increase learn weights during training
inc_ad = 0.0
inc_rec = 0.0
inc_pearson = 0.0
inc_noise = 0.0
inc_en = 0.0
inc_spec_h = 0.0
n_max = 10



#training 
BS = 1 
nb_epoch = 100
nb_batches_per_epoch = 15000

#Optimizer
dis_opt = Adagrad(lr=0.001, epsilon=1e-08)
gan_opt = Adagrad(lr=0.008, epsilon=1e-08)

save_model = 1
plot_data = 1




