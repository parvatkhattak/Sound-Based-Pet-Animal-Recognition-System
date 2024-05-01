# datset fir training:https://drive.google.com/drive/folders/1TiGKK2LxLspRfxLixaEk8_NMZ23Kwta9?usp=drive_link
import os
# Standard library for interacting with the operating system.

from tkinter import messagebox
# Import messagebox from tkinter for creating simple message boxes in the GUI.

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Disable oneDNN optimizations in TensorFlow, which can be necessary for compatibility or performance reasons.

import tkinter as tk
# Import tkinter for creating graphical user interfaces.

from tkinter import filedialog, PhotoImage
# filedialog for opening file dialog boxes to choose files, PhotoImage to handle image formats in tkinter.

import numpy as np
# Import NumPy for numerical operations and array handling, widely used in data manipulation.

import librosa
# Import librosa, a library for audio and music analysis.

import matplotlib.pyplot as plt
# Import matplotlib's pyplot for plotting graphs and visualizing data.

import pygame
# Import pygame, a library for writing video games and multimedia applications.

import tensorflow as tf
# Import TensorFlow, a framework for machine learning and neural networks.

# Disable informational messages from TensorFlow to make the output less verbose.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Settings to suppress TensorFlow's default console logs: 0 = all messages, 1 = suppress INFO, 2 = suppress INFO and WARNING, 3 = suppress all messages.

from tensorflow.keras.utils import register_keras_serializable # type: ignore
# Import register_keras_serializable to allow custom objects in Keras to be serializable, useful for saving/loading models with custom components.

@register_keras_serializable()
# Decorator to register the 'sensitivity' function as a serializable custom object in Keras.
def sensitivity(y_true, y_pred):
    # Calculate the sum of true positives: predictions that are correctly identified as positive.
    true_positive = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    # Calculate the sum of all actual positives in the ground truth.
    possible_positive = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    # Return the sensitivity as the ratio of true positives to all actual positives, with epsilon to prevent division by zero.
    return true_positive / (possible_positive + tf.keras.backend.epsilon())

@register_keras_serializable()
# Decorator to register the 'specificity' function as a serializable custom object in Keras.
def specificity(y_true, y_pred):
    # Calculate the sum of true negatives: predictions that are correctly identified as negative.
    true_negative = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    # Calculate the sum of all actual negatives in the ground truth.
    possible_negative = tf.reduce_sum(tf.round(tf.clip_by_value(1 - y_true, 0, 1)))
    # Return the specificity as the ratio of true negatives to all actual negatives, with epsilon to prevent division by zero.
    return true_negative / (possible_negative + tf.keras.backend.epsilon())

# Load your trained model with custom metrics
model = tf.keras.models.load_model('pet_sound_recognition_model.keras', custom_objects={
    'sensitivity': sensitivity,  # Provide the custom sensitivity function defined earlier.
    'specificity': specificity   # Provide the custom specificity function defined earlier.
})

# Initialize Pygame for audio playback
pygame.init()
# This line initializes all the Pygame modules that are available. It's necessary to call this before using any other Pygame functions, particularly for audio and video systems.

pygame.mixer.init()
# This line specifically initializes the Pygame mixer module, which is used for loading and playing sounds. This is crucial for handling audio operations in your application.


# Disable oneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# This line sets the environment variable 'TF_ENABLE_ONEDNN_OPTS' to '0'. 
# oneDNN (formerly known as MKL-DNN) is a library that TensorFlow can use to accelerate performance on certain hardware. 
# Setting this to '0' disables these optimizations, which might be necessary if they are causing compatibility issues or non-deterministic behaviors in some operations.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable informational messages from TensorFlow
# This line sets the 'TF_CPP_MIN_LOG_LEVEL' environment variable to '2', which configures TensorFlow to filter out informational and warning messages.
# The levels are as follows: 0 = all logs shown, 1 = filter out INFO logs, 2 = filter out INFO and WARNING logs, 3 = filter out INFO, WARNING, and ERROR logs.
# Setting it to '2' helps reduce the console output to errors only, which can be useful for keeping the output clean and more focused on critical issues.

# Load the trained model
model = tf.keras.models.load_model('pet_sound_recognition_model.keras')

import tkinter as tk

# Initialize the main window for the GUI
root = tk.Tk()
# This creates the main window by instantiating Tk. It serves as the primary container for all other GUI elements.

root.title("Pet Sound Classification")
# Sets the title of the window, which will appear in the title bar. In this case, "Pet Sound Classification".

root.geometry('600x500')
# Specifies the size of the window. Here, the window is set to be 600 pixels wide and 500 pixels tall.

root.config(bg='#121212')
# Sets the background color of the window. The color code '#121212' is a dark grey, often used for dark themes.



def preprocess_audio(file_path):
    """
    Function to preprocess audio for model prediction.

    Args:
    file_path (str): Path to the audio file to be processed.

    Returns:
    np.ndarray: A preprocessed audio spectrogram with added axes for batch and channel.
    """
    # Set parameters for the Mel spectrogram transformation.
    n_fft = 2048  # The window size for the FFT.
    hop_length = 512  # The number of samples between successive frames.
    n_mels = 128  # The number of Mel bands.
    max_pad_len = 140  # The maximum column width of the spectrogram.

    # Load the audio file as a waveform `y` with a sample rate `sr`.
    y, sr = librosa.load(file_path, sr=None)

    # Generate a Mel spectrogram from the audio signal.
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert the power spectrogram (amplitude squared) to decibel units.
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Calculate how much padding is needed to make the spectrogram width equal to `max_pad_len`.
    pad_width = max_pad_len - S_DB.shape[1]

    # Apply padding if needed, or trim the spectrogram if it's too long.
    if pad_width > 0:
        # If the spectrogram is shorter than the maximum length, pad it with zeros.
        S_DB = np.pad(S_DB, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        # If the spectrogram is longer than the maximum length, trim it.
        S_DB = S_DB[:, :max_pad_len]

    # Add a new axis at the beginning for batch size and at the end for channels (required by many ML models).
    return S_DB[np.newaxis, ..., np.newaxis]



# Load images for the GUI
cat_image = PhotoImage(file='cat.png')  # Load a cat image for display in the GUI. Update the path if necessary.
dog_image = PhotoImage(file='dog.png')  # Load a dog image for display in the GUI. Update the path if necessary.
upload_success_image = PhotoImage(file='load.png')  # Load an image that might be used to indicate successful upload or processing.

# Initialize variables for holding audio data
audio_data = None  # Variable to store the audio waveform data once loaded.
sr = None  # Variable to store the sample rate of the audio.

# Declare a global variable to hold the file path of the currently loaded audio
current_audio_path = None


def load_audio():
    global current_audio_path  # Declare the global variable to store the path of the current audio file.
    file_path = filedialog.askopenfilename()  # Open a file dialog to select an audio file.

    if file_path:  # Check if a file path was selected (i.e., the user didn't cancel the operation).
        current_audio_path = file_path  # Save the file path globally for use elsewhere in the program.

        global audio_data  # Declare audio_data as global to modify it within this function.
        audio_data = preprocess_audio(file_path)  # Process the audio file using the preprocess_audio function.

        label_result.config(text="Audio file loaded successfully!")  # Update a label to indicate the file was loaded.
        label_image.config(image=upload_success_image)  # Display an image indicating successful loading.



def play_audio():
    """Play the loaded audio file."""
    global current_audio_path  # Access the global variable storing the path of the loaded audio file.

    if current_audio_path:  # Check if there is an audio file path available (i.e., a file has been loaded).
        pygame.mixer.music.load(current_audio_path)  # Load the audio file into Pygame's mixer.
        pygame.mixer.music.play()  # Start playing the audio file using Pygame's mixer.

        label_result.config(text="Playing audio...")  # Update the GUI label to indicate that audio is playing.

        # Update the GUI image to indicate audio is playing.
        playing_image_path = "play.png"  # Path to the image indicating that audio is currently playing.
        playing_image = PhotoImage(file=playing_image_path)
        label_image.config(image=playing_image)
        label_image.photo = playing_image  # Keep a reference to the image to prevent it from being garbage collected.

    else:  # If no audio file has been loaded, inform the user.
        label_result.config(text="No audio file loaded. Please load an audio file first.")

def predict():
    global audio_data  # Ensure 'audio_data' is accessible as it may be set outside this function

    if audio_data is None:
        # If no audio data is loaded, prompt the user to load a file first.
        label_result.config(text="No audio file loaded. Please load a file first.")
    else:
        # Use the pre-loaded model to predict the class of the audio data.
        prediction = model.predict(audio_data)
        # Get the index of the maximum value which represents the predicted class.
        predicted_class = np.argmax(prediction, axis=1)

        if predicted_class[0] == 0:
            # If the predicted class is 0, update the GUI to show it's a cat.
            label_image.config(image=cat_image)
            label_result.config(text="Prediction: It's a Cat!")
        else:
            # If the predicted class is not 0, update the GUI to show it's a dog.
            label_image.config(image=dog_image)
            label_result.config(text="Prediction: It's a Dog!")


def pause_audio():
    """
    Pause the currently playing audio and display an image indicating the pause.
    """
    pygame.mixer.music.pause()  # Pause the currently playing audio using Pygame's mixer.
    
    label_result.config(text="Audio paused.")  # Update the text label on the GUI to indicate that the audio is paused.

    # Load and display an image to indicate that the audio has been paused.
    pause_image_path = "pause.png"  # Path to the image that indicates the paused state.
    pause_image = PhotoImage(file=pause_image_path)
    label_image.config(image=pause_image)
    label_image.photo = pause_image  # Keep a reference to the PhotoImage object to prevent it from being garbage collected.


def show_mel_spectrogram():
    """
    Display the Mel spectrogram of the loaded audio using librosa and matplotlib.
    """
    global current_audio_path  # Access the global variable that holds the path to the current audio file.

    if not current_audio_path:
        # If no audio file is currently loaded, show an error message.
        messagebox.showerror("Error", "No audio file loaded.")
        return  # Exit the function if there is no audio file to process.
    
    # Load the audio file at the path stored in current_audio_path.
    y, sr = librosa.load(current_audio_path, sr=None)  # Load the audio file with its original sample rate.
    # Generate a Mel spectrogram from the audio data.
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    # Create a new figure for the spectrogram.
    plt.figure(figsize=(10, 4))
    # Display the spectrogram with librosa's specshow.
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    # Add a color bar to the plot to indicate the decibel levels.
    plt.colorbar(format='%+2.0f dB')
    # Set the title of the plot.
    plt.title('Mel Spectrogram')
    # Display the plot.
    plt.show()


# Controls frame
controls_frame = tk.Frame(root, bg='#121212')
controls_frame.pack(side=tk.BOTTOM, pady=10)  # Pack at the bottom

# Widgets
btn_load = tk.Button(controls_frame, text="Load Audio", command=load_audio, bg='#333333', fg='white', font=('Helvetica', 12))
btn_load.pack(side=tk.LEFT, padx=10)

btn_play = tk.Button(controls_frame, text="Play Audio", command=play_audio, bg='#333333', fg='white', font=('Helvetica', 12))
btn_play.pack(side=tk.LEFT, padx=10)

btn_pause = tk.Button(controls_frame, text="Pause Audio", command=pause_audio, bg='#333333', fg='white', font=('Helvetica', 12))
btn_pause.pack(side=tk.LEFT, padx=10)

btn_predict = tk.Button(controls_frame, text="Predict Audio", command=predict, bg='#333333', fg='white', font=('Helvetica', 12))
btn_predict.pack(side=tk.LEFT, padx=10)

btn_mel_spectrogram = tk.Button(controls_frame, text="Show Mel Spectrogram", command=show_mel_spectrogram, bg='#333333', fg='white', font=('Helvetica', 12))
btn_mel_spectrogram.pack(side=tk.LEFT, padx=10)

# Label for displaying results or messages
label_result = tk.Label(root, text="Load an audio file to start.", bg='#121212', fg='white', font=('Helvetica', 16))
label_result.pack(pady=20)

# Label for displaying images (e.g., dog or cat images)
label_image = tk.Label(root, bg='#121212')
label_image.pack(pady=20)

# Start the GUI application
root.mainloop()

# datset fir training:https://drive.google.com/drive/folders/1TiGKK2LxLspRfxLixaEk8_NMZ23Kwta9?usp=drive_link