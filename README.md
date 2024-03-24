# LC3 coded Audio Enhancement using GAN 

## Overview

This repository contains code for audio enhancement using a **Generative Adversarial Network (GAN)**. The _codec_ folder contains the encoder and decoder required to convert clean WAV audio files to audio that contains spectral noise. The rest of the code implements a GAN model that takes MDCT coefficients of audio as input and generates a mask. This mask, when applied to the audio, enhances its quality.

## Contents

- **codec/**
  - Contains the encoder and decoder modules.
- **data/**
  - Placeholder for audio datasets.
- **scripts/**
  - Utility scripts for data processing and training.
- **README.md**
  - This file providing an overview of the project.

## Requirements

- Python 
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
   - Run the training script to train the GAN model in `main.py`


   ```bash
   python main.py --batch_size 32 --epochs 300
   ```

3. **Audio Enhancement:**
   - After training, use the trained model to enhance audio by applying the generated mask to the MDCT coefficients.

4. **Codec Usage:**
   - Utilize the encoder and decoder modules in the `codec/` folder to convert clean WAV audio files to include spectral noise.

## References

- LC3 Codec using Rust ([ninjasource](https://github.com/ninjasource/lc3-codec))
- Speech Enhancement GAN ([leftthomas](https://github.com/leftthomas/SEGAN))
- MDCT & Inverse MDCT ([dhroth](https://github.com/dhroth/pytorch-mdct)) 

## License
```
Copyright 2024 Midhun Nirmal, Avinash K B, Diljith P A, Jithu Johan Jose

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

## Contributors

- [Midhuhn Nirmal](https://github.com/MidhunNirmal)
- [Avinash K B](https://github.com/avinash-panikkan)
- [Diljith P A](https://github.com/dilji)
- [Jithu Johan Jose](https://github.com/RoyalewidCheese)