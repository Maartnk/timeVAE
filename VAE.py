import os, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore')


import sys
import numpy as np, pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from vae_conv_I_model import VariationalAutoencoderConvInterpretable as TimeVAE
import utils
import matplotlib.pyplot as plt

data_dir = './datasets/'
# ----------------------------------------------------------------------------------
# choose model
vae_type = 'timeVAE'  # vae_dense, vae_conv, timeVAE
# ----------------------------------------------------------------------------------
# read data
load_dataset = np.load("C:/Users/MMelo/Documents/School/BA Thesis/Dataset/EEG_all_epochs.npy")
valid_perc = 0.1
input_file = load_dataset
#full_train_data = utils.get_training_data(data_dir + input_file)
print(np.shape(input_file))
N, T = input_file.shape
D = 1
print('data shape:', N, T, D)

# ----------------------------------------------------------------------------------
# further split the training data into train and validation set - same thing done in forecasting task
N_train = int(N * (1 - valid_perc))
N_valid = N - N_train

# Shuffle data
np.random.shuffle(input_file)

train_data = input_file[:N_train]
valid_data = input_file[N_train:]
print("train/valid shapes: ", train_data.shape, valid_data.shape)

# ----------------------------------------------------------------------------------
# min max scale the data
scaler = utils.MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)

scaled_valid_data = scaler.transform(valid_data)
# joblib.dump(scaler, 'scaler.save')

# ----------------------------------------------------------------------------------
# instantiate the model

# latent_dim = 8
#
# if vae_type == 'vae_dense':
#     vae = VAE_Dense(seq_len=T, feat_dim=D, latent_dim=latent_dim, hidden_layer_sizes=[200, 100], )
# elif vae_type == 'vae_conv':
#     vae = VAE_Conv(seq_len=T, feat_dim=D, latent_dim=latent_dim, hidden_layer_sizes=[100, 200])
# elif vae_type == 'timeVAE':
#     vae = TimeVAE(seq_len=T, feat_dim=D, latent_dim=latent_dim, hidden_layer_sizes=[50, 100, 200],
#                   # [80, 200, 250]
#                   reconstruction_wt=3.0,
#                   # ---------------------
#                   # disable following three arguments to use the model as TimeVAE_Base. Enabling will convert to Interpretable version.
#                   # Also set use_residual_conn= False if you want to only have interpretable components, and no residual (non-interpretable) component.
#
#                   # trend_poly=2,
#                   # custom_seas = [ (6,1), (7, 1), (8,1), (9,1)] ,     # list of tuples of (num_of_seasons, len_per_season)
#                   # use_scaler = True,
#
#                   # ---------------------------
#                   use_residual_conn=True
#                   )
# else:
#     raise Exception('wut')
#
# vae.compile(optimizer=Adam())
# # vae.summary() ; sys.exit()
#
# early_stop_loss = 'loss'
# early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta=1e-1, patience=10)
# reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)
#
# vae.fit(
#     scaled_train_data,
#     batch_size=32,
#     epochs=500,
#     shuffle=True,
#     callbacks=[early_stop_callback, reduceLR],
#     verbose=1
# )
#
# # ----------------------------------------------------------------------------------
# # save model
# model_dir = './model/'
# file_pref = f'vae_{vae_type}_iter_{0}_'
# vae.save(model_dir, file_pref)
#
# # ----------------------------------------------------------------------------------
# # visually check reconstruction
# X = scaled_train_data
#
# x_decoded = vae.predict(scaled_train_data)
# print('x_decoded.shape', x_decoded.shape)
#
# ### compare original and posterior predictive (reconstructed) samples
# #utils.draw_orig_and_post_pred_sample(X, x_decoded, n=5)
#
# # # Plot the prior generated samples over different areas of the latent space
# if latent_dim == 2: utils.plot_latent_space_timeseries(vae, n=8, figsize=(20, 10))
#
# # # ----------------------------------------------------------------------------------
# # draw random prior samples
# num_samples = N_train
# # print("num_samples: ", num_samples)
#
# samples = vae.get_prior_samples(num_samples=num_samples)
#
# utils.plot_samples(samples, n=5)

# ------------------------------------------------------------

# inverse-transform scaling
#samples = scaler.inverse_transform(samples)
#print('shape of gen samples: ', samples.shape)

# ----------------------------------------------------------------------------------
# save samples
#output_dir = './outputs/'
#sample_fname = f'{vae_type}_gen_samples.npz'
#samples_fpath = os.path.join(output_dir, sample_fname)
#np.savez_compressed(samples_fpath, data=samples)

# ----------------------------------------------------------------------------------

model_dir = './model/L586.4'
file_pref = f'vae_{vae_type}_iter_{0}_'

#load model
new_vae = TimeVAE.load(model_dir, file_pref)

new_x_decoded = new_vae.predict(scaled_train_data)
# print('new_x_decoded.shape', new_x_decoded.shape)

# draw random prior samples
num_samples = N_train
# print("num_samples: ", num_samples)

samples = new_vae.get_prior_samples(num_samples=num_samples)

utils.plot_samples(samples, n=5)

# --------------------------------------------------------------------------
print(np.shape(samples))
#plot samples
for i in range(5):
    fig, ax = plt.subplots(figsize=[9, 5])
    ax.plot(np.arange(1, 513), samples[i])
    ax.yaxis.set_visible(False)
    plt.savefig(f'./outputs/sample_{i}')
    plt.close()