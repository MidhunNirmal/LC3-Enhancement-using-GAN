# Audio Enhancement using GAN and Spectral Noise

## Overview

This repository contains code for audio enhancement using a Generative Adversarial Network (GAN) and spectral noise. The `codec` folder contains the encoder and decoder required to convert clean WAV audio files to audio that contains spectral noise. The rest of the code implements a GAN model that takes MDCT coefficients of audio as input and generates a mask. This mask, when applied to the audio, enhances its quality.

## Contents

- **codec/**
  - Contains the encoder and decoder modules.
- **gan/**
  - Implements the GAN model for audio enhancement.
- **data/**
  - Placeholder for audio datasets.
- **scripts/**
  - Utility scripts for data processing and training.
- **README.md**
  - This file providing an overview of the project.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Librosa
- matplotlib
- tqdm

<!-- You can install the required Python packages using the following command:
```bash
pip install -r requirements.txt
``` -->

## Usage

1. **Data Preparation:**
   - Place your audio datasets in the `data/` directory.
   - Use the provided scripts in the `scripts/` directory for data preprocessing if needed.

2. **Training the GAN:**
   - Run the training script located in the `gan/` directory to train the GAN model.
   ```bash
   cd gan/
   python main.py --batch_size 32 --epochs 300
   ```

3. **Audio Enhancement:**
   - After training, use the trained model to enhance audio by applying the generated mask to the MDCT coefficients.

4. **Codec Usage:**
   - Utilize the encoder and decoder modules in the `codec/` folder to convert clean WAV audio files to include spectral noise.

## References

Include any references or citations to papers, articles, or libraries that you have used for this project.

## License

Specify the license under which your project is distributed. For example:
```
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
```

## Contributors

- [Midhuhn Nirmal](https://github.com/MidhunNirmal)
- [Avinash K B](https://github.com/avinash-panikkan)

Feel free to add more sections or details based on your project's specific requirements and structure. Good luck with your audio enhancement project! 🎶✨