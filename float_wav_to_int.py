# import numpy as np

# def float_to_int_wav(float_wav, bit_depth=16):
#     # Scale the float waveform to the range [-1, 1]
#     float_wav = np.clip(float_wav, -1.0, 1.0)  # Clip values outside [-1, 1] range
#     scaled_wav = float_wav * (2**(bit_depth - 1) - 1)

#     # Convert the scaled waveform to integer type
#     int_wav = scaled_wav.astype(np.int16)  # Use np.int32 for 32-bit integer wav

#     return int_wav


import numpy as np
from scipy.io import wavfile

# Load the float waveform from the WAV file
file_path = 'enhanced_sample.wav'  # Update the file path accordingly
sample_rate, float_wav = wavfile.read(file_path)

# Normalize the float waveform to the range [-1, 1]
float_wav = float_wav / (2**15)  # Assuming a 16-bit WAV file, adjust if needed

# Scale and convert to integer waveform
int_wav = np.clip(float_wav, -1.0, 1.0) * (2**15 - 1)
int_wav = int_wav.astype(np.int16)  # Convert to 16-bit integer

# Save the integer waveform to a new WAV file
output_file_path = 'output_file.wav'  # Specify the output file path
wavfile.write(output_file_path, sample_rate, int_wav)

print("Conversion complete. Integer WAV file saved:", output_file_path)