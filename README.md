# CIFAR-10 Image Denoising Project

This project demonstrates how to remove noise from images using a Convolutional Autoencoder. It uses the CIFAR-10 dataset, adding Gaussian noise to simulate real-world noisy data, and aims to reconstruct clean images using a deep learning model built with TensorFlow.

## Key Features:
- Convolutional Autoencoder for denoising images.
- CIFAR-10 dataset for training and evaluation.
- Visualization of noisy and reconstructed images.


## Project Overview

This project demonstrates an image denoising autoencoder model applied to the CIFAR-10 dataset. The goal is to remove noise from images using a convolutional neural network (CNN) model. The noisy images are created by adding random Gaussian noise to the original CIFAR-10 dataset images. The model is trained to reconstruct the original, clean images from the noisy ones.

## Dataset

The CIFAR-10 dataset is used in this project, which consists of 60,000 32x32 color images in 10 different classes (e.g., airplane, automobile, bird, etc.). The dataset is automatically downloaded from the `keras.datasets` module.

### Dataset Structure:
- **Training set**: 50,000 images
- **Test set**: 10,000 images

### Labels:
- The labels range from 0 to 9, representing the following classes:
  - 0: Airplane
  - 1: Automobile
  - 2: Bird
  - 3: Cat
  - 4: Deer
  - 5: Dog
  - 6: Frog
  - 7: Horse
  - 8: Ship
  - 9: Truck

## Steps in the Project

1. **Loading the CIFAR-10 Dataset**:
    - The dataset is loaded using `datasets.cifar10.load_data()` and split into training and testing sets.

2. **Data Visualization**:
    - Random images from the training dataset are displayed with their corresponding class labels to better understand the dataset.

3. **Image Normalization**:
    - Both training and testing images are normalized by dividing pixel values by 255 to scale them between 0 and 1.

4. **Adding Noise (Denoising)**:
    - Gaussian noise is added to the training and test images to simulate noisy data. The goal of the model is to reconstruct the original, clean images from these noisy versions.

5. **Training and Validation Split**:
    - A portion of the noisy training dataset is split into a validation set to monitor model performance during training.

6. **Model Architecture**:
    - A Convolutional Autoencoder is built using `TensorFlow` and `Keras`. The model consists of an encoding and decoding phase:
        - **Encoder**: Uses convolutional layers, batch normalization, and max pooling to down-sample the noisy input images and extract features.
        - **Decoder**: Uses transposed convolutional layers to up-sample the encoded feature maps and reconstruct the clean images.
    - The output is a 32x32x3 image, similar to the original input.

7. **Model Compilation**:
    - The model is compiled using the Adam optimizer, mean squared error (MSE) loss, and accuracy as the evaluation metric.

8. **Training**:
    - The model is trained for 50 epochs with a batch size of 64, using noisy images as inputs and clean images as the ground truth.

9. **Results Visualization**:
    - After training, the model's predictions on noisy images are visualized alongside the original clean images to evaluate the denoising performance.

## Key Files and Functions

### `afficher_images(images, labels)`
- Displays a set of images along with their corresponding labels.

### `normaliser_images(images)`
- Normalizes the images by scaling the pixel values between 0 and 1.

### `ajouter_bruit(images)`
- Adds Gaussian noise to the images to simulate noisy data.

### `model()`
- Defines the architecture of the convolutional autoencoder with an encoder-decoder structure for denoising.

### `history = denoiser.fit()`
- Trains the model on noisy images with the goal of reconstructing clean images.

## Dependencies

- `tensorflow`
- `numpy`
- `matplotlib`
- `sklearn`
  
Install the necessary dependencies with:
```
pip install tensorflow matplotlib scikit-learn
```

## How to Run

1. Clone the project repository.
2. Install the required libraries listed above.
3. Run the script to load the CIFAR-10 dataset, add noise, and train the denoising autoencoder model.
4. The training history and denoised images will be displayed as part of the script's output.

## Expected Output

- Loss and accuracy plots across training epochs.
- Visualizations of noisy images alongside their clean counterparts to demonstrate the denoising capability of the model.
