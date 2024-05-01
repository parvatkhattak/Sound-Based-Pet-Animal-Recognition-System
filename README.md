# Sound-Based Pet Animal Recognition System

This project aims to develop a sound-based pet animal recognition system for tracking pets within a home environment. The system utilizes a combination of Mel Spectrogram for feature extraction and a two-dimensional convolutional neural network (2DCNN) for classification, effectively distinguishing between the sounds of cats and dogs.

## Features

- Collects diverse audio samples of cats and dogs from various sources.
- Preprocesses audio signals by denoising and normalizing.
- Extracts features using Mel Spectrograms.
- Trains a 2DCNN model for sound classification.
- Evaluates model performance using accuracy, sensitivity, specificity, and loss metrics.
- Integrates the model into a graphical user interface (GUI) for real-time usage.
- Provides insights into the model's performance through visual representations such as confusion matrices and spectrograms.
- Allows for future enhancements like expanding the dataset, real-time processing, integration with IoT devices, and improving the user interface.

## Requirements

- Python 3.8 or higher
- TensorFlow 2.x
- Librosa
- Other dependencies as specified in the system specifications section of the paper

## Getting Started
- Clone the repository

