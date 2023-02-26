import numpy as np
from vae_conv_I_model import VariationalAutoencoderConvInterpretable as TimeVAE
import utils
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import mne

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

model_dir = './model/L586.4'
file_pref = f'vae_{vae_type}_iter_{0}_'

#load model
new_vae = TimeVAE.load(model_dir, file_pref)

new_x_decoded = new_vae.predict(scaled_train_data)
# print('new_x_decoded.shape', new_x_decoded.shape)

# draw random prior samples
num_samples = 4514
# print("num_samples: ", num_samples)

samples = new_vae.get_prior_samples(num_samples=num_samples)
# np.save("C:/Users/MMelo/Documents/School/BA Thesis/VAE/timeVAE-main/timeVAE-main/samples/vae_samples",samples)

utils.plot_samples(samples, n=5, text='TimeVAE')