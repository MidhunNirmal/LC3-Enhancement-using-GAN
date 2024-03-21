import argparse
import os

import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import data_preprocess
import numpy as np

from data_preprocess import sample_rate
from model import Generator, Discriminator
from utils import AudioDataset, emphasis



def stft_rmse_loss(clean_batch, lossy_batch, n_fft=2048, hop_length=512):
    """
    Compute the STFT RMSE loss between a batch of clean audio signals and a batch of lossy audio signals.
    
    Args:
    - clean_batch (np.ndarray): Batch of clean audio signals with shape (batch_size, 1, signal_length).
    - lossy_batch (np.ndarray): Batch of lossy audio signals with shape (batch_size, 1, signal_length).
    - n_fft (int): Number of FFT points for STFT computation.
    - hop_length (int): Hop length (stride) in samples for STFT computation.
    
    Returns:
    - float: Mean RMSE loss across the batch.
    """
    assert clean_batch.shape == lossy_batch.shape, "Clean and lossy batches must have the same shape"
    
    batch_size, _, signal_length = clean_batch.shape
    total_loss = 0.0
    
    # Compute STFT and RMSE loss for each signal in the batch
    for i in range(50):
        output1 = clean_batch.cpu().detach().numpy()  # Extract clean audio signal
        output2 = lossy_batch.cpu().detach().numpy()  # Extract clean audio signal
        # lossy_audio = lossy_batch[i, 0, :]  # Extract lossy audio signal
        
        # Compute STFT for clean and lossy signals
        clean_mdct = data_preprocess.mdct(output1[i][0])
        lossy_mdct = data_preprocess.mdct(output2[i][0])
        
        # Compute magnitude spectrograms and RMSE loss
        clean_mag = np.abs(clean_mdct)
        lossy_mag = np.abs(lossy_mdct)
        rmse = np.sqrt(np.mean((clean_mag - lossy_mag)**2))
        
        total_loss += rmse  # Accumulate RMSE loss
    
    mean_loss = total_loss / batch_size  # Compute mean RMSE loss across the batch
    return mean_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=86, type=int, help='train epochs number')

    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    # load data
    print('loading data...')
    train_dataset = AudioDataset(data_type='train')
    test_dataset = AudioDataset(data_type='test')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    
    
    
    # generate reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE)

    # create D and G instances
    discriminator = Discriminator()
    generator = Generator()
    # if torch.cuda.is_available():
    #     discriminator.cuda()
    #     generator.cuda()
    #     ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)
    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))
    # optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)
    
    
    for epoch in range(50):
        train_bar = tqdm(train_data_loader)
        for train_batch, train_clean, train_noisy in train_bar:
            
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # latent vector - normal distribution
            # output1 = train_clean.cpu().detach().numpy()
            
            
                
                # print(data_preprocess.mdct(output1[i][0]))
            mdct_rmse = stft_rmse_loss(train_clean,train_noisy)
            print("count",epoch, mdct_rmse)
            
                
                
                

