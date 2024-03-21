import os

import librosa
import numpy as np
from tqdm import tqdm
import torch
import zaf
import numpy
import scipy

clean_train_folder = 'data/clean_trainset_wav'
noisy_train_folder = 'data/noisy_trainset_wav'
clean_test_folder = 'data/clean_testset_wav/clean_testset_wav'
noisy_test_folder = 'data/noisy_testset_wav/noisy_testset_wav'
serialized_train_folder = 'data/serialized_train_data'
serialized_test_folder = 'data/serialized_test_data'
window_size = 2 ** 14  # about 1 second of samples
sample_rate = 16000


def slice_signal(file, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(file, sr=sample_rate)
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize(data_type):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    stride = 0.5

    if data_type == 'train':
        clean_folder = clean_train_folder
        noisy_folder = noisy_train_folder
        serialized_folder = serialized_train_folder
    else:
        clean_folder = clean_test_folder
        noisy_folder = noisy_test_folder
        serialized_folder = serialized_test_folder
    if not os.path.exists(serialized_folder):
        os.makedirs(serialized_folder)

    # walk through the path, slice the audio file, and save the serialized result
    for root, dirs, files in os.walk(clean_folder):
        if len(files) == 0:
            continue
        for filename in tqdm(files, desc='Serialize and down-sample {} audios'.format(data_type)):
            clean_file = os.path.join(clean_folder, filename)
            noisy_file = os.path.join(noisy_folder, filename)
            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_file, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_file, window_size, stride, sample_rate)
            # serialize - file format goes [original_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(serialized_folder, '{}_{}'.format(filename, idx)), arr=pair)


def data_verify(data_type):
    """
    Verifies the length of each data after pre-process.
    """
    if data_type == 'train':
        serialized_folder = serialized_train_folder
    else:
        serialized_folder = serialized_test_folder

    for root, dirs, files in os.walk(serialized_folder):
        for filename in tqdm(files, desc='Verify serialized {} audios'.format(data_type)):
            data_pair = np.load(os.path.join(root, filename))
            if data_pair.shape[1] != window_size:
                print('Snippet length not {} : {} instead'.format(window_size, data_pair.shape[1]))
                break
            
            






# def compute_mdct(signal, window_size=512, hop_size=256):
#     audio_signal, sampling_frequency = signal
#     audio_signal = np.mean(audio_signal, 1)

#             # Compute the Kaiser-Bessel-derived (KBD) window as used in the AC-3 audio coding format
#     window_length = 512
#     alpha_value = 5
#     window_function = np.kaiser(int(window_length/2)+1, alpha_value*np.pi)
#     window_function2 = np.cumsum(window_function[1:int(window_length/2)])
#     window_function = np.sqrt(np.concatenate((window_function2, window_function2[int(window_length/2)::-1]))
#                                     /np.sum(window_function))
    
#     mdct = zaf.mdct(signal, window_function)
    
   
#     return mdct










def cmdct(x, odd=True):
    """ Calculate complex MDCT/MCLT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:True.

    Returns
    -------
    out : array_like
        The output signal

    """
    N = len(x) // 2
    n0 = (N + 1) / 2
    if odd:
        outlen = N
        pre_twiddle = numpy.exp(-1j * numpy.pi * numpy.arange(N * 2) / (N * 2))
        offset = 0.5
    else:
        outlen = N + 1
        pre_twiddle = 1.0
        offset = 0.0

    post_twiddle = numpy.exp(
        -1j * numpy.pi * n0 * (numpy.arange(outlen) + offset) / N
    )

    X = scipy.fftpack.fft(x * pre_twiddle)[:outlen]

    if not odd:
        X[0] *= numpy.sqrt(0.5)
        X[-1] *= numpy.sqrt(0.5)

    return X * post_twiddle * numpy.sqrt(1 / N)



def mdct(x, odd=True):
    """ Calculate modified discrete cosine transform of input signal

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:True.

    Returns
    -------
    out : array_like
        The output signal

    """
    return numpy.real(cmdct(x, odd=odd)) * numpy.sqrt(2)



























# import numpy as np
# import librosa

# def stft_rmse_loss(clean_batch, lossy_batch, n_fft=2048, hop_length=512):
#     """
#     Compute the STFT RMSE loss between a batch of clean audio signals and a batch of lossy audio signals.
    
#     Args:
#     - clean_batch (np.ndarray): Batch of clean audio signals with shape (batch_size, 1, signal_length).
#     - lossy_batch (np.ndarray): Batch of lossy audio signals with shape (batch_size, 1, signal_length).
#     - n_fft (int): Number of FFT points for STFT computation.
#     - hop_length (int): Hop length (stride) in samples for STFT computation.
    
#     Returns:
#     - float: Mean RMSE loss across the batch.
#     """
#     assert clean_batch.shape == lossy_batch.shape, "Clean and lossy batches must have the same shape"
    
#     batch_size, _, signal_length = clean_batch.shape
#     total_loss = 0.0
    
#     # Compute STFT and RMSE loss for each signal in the batch
#     for i in range(batch_size):
#         clean_audio = clean_batch[i, 0, :]  # Extract clean audio signal
#         lossy_audio = lossy_batch[i, 0, :]  # Extract lossy audio signal
        
#         # Compute STFT for clean and lossy signals
#         clean_stft = librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length)
#         lossy_stft = librosa.stft(lossy_audio, n_fft=n_fft, hop_length=hop_length)
        
#         # Compute magnitude spectrograms and RMSE loss
#         clean_mag = np.abs(clean_stft)
#         lossy_mag = np.abs(lossy_stft)
#         rmse = np.sqrt(np.mean((clean_mag - lossy_mag)**2))
        
#         total_loss += rmse  # Accumulate RMSE loss
    
#     mean_loss = total_loss / batch_size  # Compute mean RMSE loss across the batch
#     return mean_loss

# # Example usage
# # Generate random batch of clean and lossy audio signals
# # batch_size = 32
# # signal_length = 16384
# # clean_batch = np.random.randn(batch_size, 1, signal_length)
# # lossy_batch = np.random.randn(batch_size, 1, signal_length) 









if __name__ == '__main__':
    process_and_serialize('train')
    data_verify('train')
    process_and_serialize('test')
    data_verify('test')
