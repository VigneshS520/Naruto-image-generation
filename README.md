# NarutoGAN Image Generation

## Overview
This project implements a Generative Adversarial Network (GAN) to generate images of characters from the Naruto anime series. The GAN is trained on a dataset of Naruto character images and can generate new, realistic-looking images of characters that do not exist in the original dataset.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook (optional, for running the provided notebooks)

## Installation
1. Clone the repository:
   git clone https://github.com/VigneshS520/NarutoGAN.git
2. Install the required dependencies:
pip install -r requirements.txt


## Usage
1. Data Preparation:
- Download a dataset of Naruto character images (e.g., from Kaggle or other sources).
- Preprocess the dataset as needed (resizing, normalization, etc.).

2. Training:
- Run the training script:
  ```
  python train.py --data_path data/naruto_dataset --epochs 100 --batch_size 32
  ```
- Adjust hyperparameters such as epochs, batch size, learning rate, etc., as needed.

3. Generating Images:
- Use the trained model to generate new images:
  ```
  python generate_images.py --model_path checkpoints/generator.pth --num_images 10 --output_dir generated_images/
  ```
- Specify the path to the trained generator model, the number of images to generate, and the output directory.

4. Evaluation:
- Evaluate the generated images qualitatively and quantitatively using metrics like Inception Score, FID score, etc.


