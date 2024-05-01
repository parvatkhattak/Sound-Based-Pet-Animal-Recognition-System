# datset fir training:https://drive.google.com/drive/folders/1TiGKK2LxLspRfxLixaEk8_NMZ23Kwta9?usp=drive_link
import os
# Set an environment variable to disable oneDNN optimizations in TensorFlow for potential compatibility or performance reasons.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
# Import NumPy, a fundamental package for scientific computing with Python, used for working with arrays and numerical operations.

import librosa
# Import librosa, a Python library for analyzing audio and music. It provides the building blocks necessary to create music information retrieval systems.

import tensorflow as tf
# Import TensorFlow, an open-source library for numerical computation that makes machine learning faster and easier.

# Disable informational messages from TensorFlow to make the output less verbose, which is useful for reducing log clutter when you are debugging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages are logged, 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed

# Define a function to calculate sensitivity, also known as true positive rate or recall.
def sensitivity(y_true, y_pred):
    # Calculate the true positives: Predictions and actual values that are both 1.
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    # Calculate the possible positives: All actual values that are 1.
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    # Return the ratio of true positives to possible positives, adding a small epsilon to avoid division by zero.
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

# Define a function to calculate specificity, also known as true negative rate.
def specificity(y_true, y_pred):
    # Calculate the true negatives: Predictions and actual values that are both 0.
    true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    # Calculate the possible negatives: All actual values that are 0.
    possible_negatives = tf.reduce_sum(tf.round(tf.clip_by_value(1 - y_true, 0, 1)))
    # Return the ratio of true negatives to possible negatives, adding a small epsilon to avoid division by zero.
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

# Define the path to the saved model.
model_path = 'pet_sound_recognition_model.keras'

# Load the pre-trained model using TensorFlow's Keras API.
# The 'custom_objects' parameter is used to specify additional functions or classes that Keras needs to know about because of custom objects or layers, in this case, the custom metrics.
model = tf.keras.models.load_model(model_path, custom_objects={
    'sensitivity': sensitivity,  # The sensitivity function defined previously.
    'specificity': specificity   # The specificity function defined previously.
})


def preprocess_data(audio_file_path, n_fft=2048, hop_length=512, n_mels=128, max_pad_len=140):
    # Load an audio file as a floating point time series using librosa.
    y, sr = librosa.load(audio_file_path, sr=None)

    # Compute a mel-scaled spectrogram, which is a representation of the short-term power spectrum of a sound.
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert the mel-spectrogram (power scale) to decibel (dB) scale, which is a logarithmic scale.
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Calculate how much padding is needed to make all spectrograms have the same length.
    pad_width = max_pad_len - S_DB.shape[1]

    # If padding is needed (spectrogram is shorter than the max length), pad the spectrogram.
    if pad_width > 0:
        S_DB = np.pad(S_DB, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # If no padding is needed (spectrogram is longer than the max length), trim the spectrogram.
    else:
        S_DB = S_DB[:, :max_pad_len]

    # Add a new axis to the array to make it suitable for TensorFlow/Keras which expects input as batches of data.
    return S_DB[np.newaxis, ..., np.newaxis]


# Function to predict the class of a sound and evaluate the performance of the model.
def evaluate_model(model, audio_file_path, true_label):
    # Preprocess the audio file to transform it into a format suitable for the model.
    preprocessed_data = preprocess_data(audio_file_path)
    
    # Use the model to predict the class of the processed audio data.
    prediction = model.predict(preprocessed_data)
    
    # Determine the class with the highest probability from the prediction.
    predicted_class_index = np.argmax(prediction, axis=1)
    
    # Map numerical class indices to string labels.
    class_labels = {0: 'Cat', 1: 'Dog'}
    
    # Print the predicted and actual classes to provide a simple form of evaluation.
    print(f"Predicted class: {class_labels[predicted_class_index[0]]}, Actual class: {class_labels[true_label]}")
    
    # Convert the integer labels to one-hot encoded format to use in custom metrics calculations.
    y_true = tf.one_hot([true_label], depth=2)
    y_pred = tf.one_hot(predicted_class_index, depth=2)
    
    # Calculate sensitivity, specificity, and accuracy to evaluate the model.
    sensitivity_score = sensitivity(y_true, y_pred).numpy()  # Calculate sensitivity using the custom function.
    specificity_score = specificity(y_true, y_pred).numpy()  # Calculate specificity using the custom function.
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1)), tf.float32)).numpy()
    
    # Print the calculated metrics.
    print(f"Sensitivity: {sensitivity_score}")
    print(f"Specificity: {specificity_score}")
    print(f"Accuracy: {accuracy}")

# Test the model using a sample audio file path and a true label (0 for Cat, 1 for Dog).
audio_file_path = r"C:\Users\sweet\Downloads\dsp_project\test\test2.wav"
true_label = 0  # True label for the audio file, indicating it's a cat.
evaluate_model(model, audio_file_path, true_label)
