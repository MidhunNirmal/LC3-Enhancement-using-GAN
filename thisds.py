import numpy as np
import wave

# Load float format audio
float_audio, sr = wave.read('enhanced_sample.wav')

# Convert to PCM encoding (integer)
pcm_audio = (float_audio * 32767).astype('int16')

# Save as WAV file
with wave.open('output_pcm.wav', 'w') as wav_file:
    wav_file.setparams((1, 2, sr, len(pcm_audio), 'NONE', 'not compressed'))
    wav_file.writeframes(pcm_audio.tobytes())
