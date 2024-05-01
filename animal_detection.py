# datset fir training:https://drive.google.com/drive/folders/1TiGKK2LxLspRfxLixaEk8_NMZ23Kwta9?usp=drive_link
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for TensorFlow.

import numpy as np  # Import NumPy for numerical operations.
import librosa  # Import librosa for audio processing.
import librosa.display  # Import librosa.display for displaying audio information.
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting graphs.
import tensorflow as tf # Import TensorFlow for building and training neural networks.
from sklearn.metrics import confusion_matrix, classification_report
# Import `confusion_matrix` and `classification_report` from `sklearn.metrics`.
# - `confusion_matrix`: This function computes the confusion matrix to evaluate the accuracy of a classification.
#   It helps to visualize the performance of an algorithm by comparing the actual target values with those predicted by the model.
# - `classification_report`: This function builds a text report showing the main classification metrics including precision,
#   recall, f1-score, and supports on a per-class basis. This is crucial for understanding the model's performance for each class.

import seaborn as sns
# Import the `seaborn` module, which is a Python data visualization library based on matplotlib.
# It provides a high-level interface for drawing attractive and informative statistical graphics.
# This import is typically used for making the confusion matrix more visually appealing with color-encoded matrices.

# Disable informational messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages are logged, 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed

from tensorflow.keras.models import Sequential  # type: ignore # Import Sequential from tensorflow.keras.models to build a linear stack of layers.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore # Import necessary layers for building the model.
from tensorflow.keras.utils import to_categorical  # type: ignore # Import to_categorical for converting class vectors to binary class matrices.
from tensorflow.keras.layers import Input  # type: ignore # Import Input to create input layer for the model.
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, CategoricalAccuracy  # type: ignore # Import metrics for evaluating the model.
from sklearn.model_selection import train_test_split  # Import train_test_split to easily split data into training and test sets.


def ensure_dir(file_path):
    """
    Ensure directory exists, and if not, create it.
    This function checks if a directory exists at the specified file path.
    If the directory does not exist, it creates the directory including any necessary parent directories.
    
    Args:
    file_path (str): The path of the directory to check and potentially create.
    """
    if not os.path.exists(file_path):  # Check if the directory at the given file_path does not exist.
        os.makedirs(file_path)  # Create the directory at file_path including any necessary parent directories.

def preprocess_data(audio_directory, npy_output_directory, img_output_directory, n_fft=2048, hop_length=512, n_mels=128, max_pad_len=140):
    """
    Extract Mel Spectrograms, pad them to a uniform length, save numpy arrays in one directory and images in another.
    Parameters:
    - audio_directory: Directory containing audio files.
    - npy_output_directory: Output directory for numpy files.
    - img_output_directory: Output directory for images.
    - n_fft: FFT window size for Mel Spectrogram.
    - hop_length: Hop length for Mel Spectrogram.
    - n_mels: Number of Mel bands.
    - max_pad_len: Maximum padding length for spectrogram consistency.
    """
    audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]  # List all .wav files in the audio_directory.
    ensure_dir(npy_output_directory)  # Ensure the numpy output directory exists.
    ensure_dir(img_output_directory)  # Ensure the image output directory exists.
    for file in audio_files:  # Iterate through each audio file.
        y, sr = librosa.load(os.path.join(audio_directory, file), sr=None)  # Load the audio file with librosa.
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)  # Generate the Mel spectrogram.
        S_DB = librosa.power_to_db(S, ref=np.max)  # Convert the Mel spectrogram to decibel units.

        # Pad or truncate the columns of spectrogram to max_pad_len
        pad_width = max_pad_len - S_DB.shape[1]  # Calculate the padding width needed.
        if pad_width > 0:
            S_DB_padded = np.pad(S_DB, pad_width=((0, 0), (0, pad_width)), mode='constant')  # Pad the spectrogram if it is too short.
        else:
            S_DB_padded = S_DB[:, :max_pad_len]  # Truncate the spectrogram if it is too long.

        # Save the spectrogram as a numpy array
        np_file_path = os.path.join(npy_output_directory, file.split('.')[0] + '.npy')  # Define the path for saving the numpy file.
        np.save(np_file_path, S_DB_padded)  # Save the padded spectrogram as a numpy array.

        # Save the spectrogram as an image
        plt.figure(figsize=(10, 4))  # Set the figure size for the plot.
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')  # Display the spectrogram.
        plt.title('Mel Spectrogram')  # Title for the plot.
        plt.colorbar(format='%+2.0f dB')  # Add a color bar to the plot to indicate the decibel level.
        img_file_path = os.path.join(img_output_directory, file.split('.')[0] + '.png')  # Define the path for saving the image.
        plt.savefig(img_file_path)  # Save the plot as an image file.
        plt.close()  # Close the plot to free up memory.
        print(f"{file}: Numpy file saved to {np_file_path}, image saved to {img_file_path}.")  # Log the save locations of the numpy array and image.

def load_data(data_directory):
    """
    Load spectrogram data from numpy files and prepare for training.
    - data_directory: Directory containing numpy files
    This function reads spectrogram data saved as numpy arrays from a specified directory,
    extracts the class labels from the filenames, and prepares the data for model training by
    splitting it into training and testing datasets.
    """
    files = [f for f in os.listdir(data_directory) if f.endswith('.npy')]  # List all files ending with '.npy' in the given directory.
    X, y = [], []  # Initialize empty lists for storing spectrograms (X) and labels (y).
    for file in files:  # Loop through each file in the list.
        mel_spectrogram = np.load(os.path.join(data_directory, file))  # Load the numpy file containing the spectrogram.
        label = 0 if 'cat' in file else 1  # Assign label '0' for cats and '1' for dogs based on the file name.
        X.append(mel_spectrogram)  # Append the spectrogram to the X list.
        y.append(label)  # Append the label to the y list.

    X = np.array(X)  # Convert the list of spectrograms into a numpy array for processing.
    X = X[..., np.newaxis]  # Add a channel dimension to the array to fit the input requirements of convolutional layers.
    y = to_categorical(y)  # Convert the list of labels into a binary class matrix for use with categorical crossentropy.
    return train_test_split(X, y, test_size=0.2, random_state=42)  # Split the dataset into training and testing sets, and return them.




def sensitivity(y_true, y_pred):
    """
    Calculate the sensitivity (also called the true positive rate or recall) for binary classification.
    Args:
    y_true (tensor): True labels, one-hot encoded if multiclass.
    y_pred (tensor): Predicted probabilities or binary predictions.
    Returns:
    float: The sensitivity of the model.
    """
    # Calculate True Positives: predict positive and it's true
    true_positive = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    # Calculate possible positives: all actual positive cases in y_true
    possible_positive = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positive / (possible_positive + tf.keras.backend.epsilon())  # Return sensitivity. Add a small epsilon to avoid division by zero.

def specificity(y_true, y_pred):
    """
    Calculate the specificity (also called the true negative rate) for binary classification.
    Args:
    y_true (tensor): True labels, one-hot encoded if multiclass.
    y_pred (tensor): Predicted probabilities or binary predictions.
    Returns:
    float: The specificity of the model.
    """
    # Calculate True Negatives: predict negative and it's true
    true_negative = tf.reduce_sum(tf.round(tf.clip_by_value((1-y_true) * (1-y_pred), 0, 1)))
    # Calculate possible negatives: all actual negative cases in y_true
    possible_negative = tf.reduce_sum(tf.round(tf.clip_by_value(1-y_true, 0, 1)))
    return true_negative / (possible_negative + tf.keras.backend.epsilon())  # Return specificity. Add a small epsilon to avoid division by zero.

def build_model(input_shape, num_classes):
    """
    Build a 2D Convolutional Neural Network (CNN) model for image or spectrogram classification.
    
    Args:
    input_shape (tuple): The shape of the input data, which includes the height, width, and channels.
    num_classes (int): The number of distinct classes for the classification task.

    Returns:
    model (tensorflow.keras.models.Sequential): The compiled CNN model ready for training.
    """
    model = Sequential([
        Input(shape=input_shape),  # Start model with an Input layer, specifying the expected input data shape
        Conv2D(32, (3, 3), activation='relu'),  # First convolution layer with 32 filters of size 3x3 and ReLU activation
        MaxPooling2D((2, 2)),  # First max pooling layer with pool size of 2x2 to reduce spatial dimensions
        Conv2D(64, (3, 3), activation='relu'),  # Second convolution layer with 64 filters of size 3x3 and ReLU activation
        MaxPooling2D((2, 2)),  # Second max pooling layer with pool size of 2x2
        Flatten(),  # Flatten the 3D outputs to 1D before passing them to the dense layers
        Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
        Dropout(0.5),  # Dropout layer to prevent overfitting, dropping out 50% of the units
        Dense(num_classes, activation='softmax')  # Output layer with a softmax activation to classify the inputs into multiple classes
    ])
    model.compile(optimizer='adam',  # Compile the model using the Adam optimizer
                  loss='categorical_crossentropy',  # Use categorical crossentropy as the loss function for a multi-class classification
                  metrics=[
                      'accuracy',  # Include accuracy as a metric for evaluation
                      sensitivity,  # Custom metric function for sensitivity
                      specificity   # Custom metric function for specificity
                  ])
    return model  # Return the compiled model



# Directory paths
audio_directory = r"C:\Users\sweet\Downloads\dsp_project\training"  # Path to the directory where audio files for training are stored.
npy_output_directory = r"C:\Users\sweet\Downloads\dsp_project\npy_files"  # Path to the directory where numpy files (spectrograms) will be saved.
img_output_directory = r"C:\Users\sweet\Downloads\dsp_project\images"  # Path to the directory where spectrogram images will be saved.


# Process audio files to create Mel Spectrograms and save them
preprocess_data(audio_directory, npy_output_directory, img_output_directory)
# This function call takes audio files from 'audio_directory', processes each to extract Mel spectrograms,
# saves the spectrograms as numpy arrays in 'npy_output_directory', and as images in 'img_output_directory'.

# Load data for training
X_train, X_test, y_train, y_test = load_data(npy_output_directory)
# This function call loads the processed spectrogram numpy files from 'npy_output_directory',
# splits them into training and testing datasets. 'X' contains the spectrogram data, and 'y' contains the labels.

# Define class labels for the confusion matrix and reports
class_labels = ['Cat', 'Dog']  # Modify according to your actual classes

# Configure and train the model
input_shape = X_train.shape[1:]  # Determine the shape of input data excluding the sample axis
num_classes = y_train.shape[1]   # Determine the number of classes from the shape of the labels
model = build_model(input_shape, num_classes)
# This function constructs the model architecture, specifying the input shape and number of output classes.

history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)
# This line trains the model using the training data with a batch size of 32, over 15 epochs, and also evaluates
# the model on a validation set during training to monitor performance on unseen data.
# Train the model and save training history

# Plotting training and validation loss
plt.figure(figsize=(10, 4))  # Create a new figure for plotting, set size to 10x4 inches for better visibility
plt.plot(history.history['loss'], label='Training Loss')  # Plot the training loss retrieved from the history object
plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot the validation loss retrieved from the history object
plt.title('Loss per Epoch')  # Set the title of the plot
plt.xlabel('Epoch')  # Label the x-axis as 'Epoch'
plt.ylabel('Loss')  # Label the y-axis as 'Loss'
plt.legend()  # Add a legend to the plot to identify which line corresponds to training or validation
plt.savefig('Loss_per_Epoch.png')  # Save the plot to a file named 'Loss_per_Epoch.png'
plt.close()  # Close the plot to free up memory and avoid displaying it in a non-interactive environment

# Plotting training and validation accuracy
plt.figure(figsize=(10, 4))  # Create another new figure for plotting accuracy, also set size to 10x4 inches
plt.plot(history.history['accuracy'], label='Training Accuracy')  # Plot the training accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot the validation accuracy
plt.title('Accuracy per Epoch')  # Set the title of the plot
plt.xlabel('Epoch')  # Label the x-axis as 'Epoch'
plt.ylabel('Accuracy')  # Label the y-axis as 'Accuracy'
plt.legend()  # Include a legend to distinguish between training and validation lines
plt.savefig('Accuracy_per_Epoch.png')  # Save the plot to a file named 'Accuracy_per_Epoch.png'
plt.close()  # Close the plot, especially useful in scripts where multiple plots might be generated


# Evaluate the model
results = model.evaluate(X_test, y_test)
# This line evaluates the trained model's performance using the test dataset.
# The `evaluate` function returns the loss value and metrics (like accuracy) specified during model compilation.
# `results` will contain these values, which typically include loss and accuracy when using typical classification metrics.

# Predict classes with the test set
y_pred = model.predict(X_test)
# This line uses the trained model to make predictions on the test dataset.
# The `predict` function outputs the raw prediction results from the model, typically probability scores for each class
# in classification tasks. For models with softmax activation in the output layer, these can be interpreted as
# probability distributions over the classes.

y_pred_classes = np.argmax(y_pred, axis=1)
# This line converts the probabilities from `y_pred` to class indices.
# `np.argmax` finds the index of the highest probability along axis 1 (i.e., among classes),
# effectively translating each set of class probabilities in `y_pred` into a single class prediction.

y_true_classes = np.argmax(y_test, axis=1)
# This line extracts the true class indices from the test labels for comparison.
# Assuming `y_test` contains one-hot encoded classes, `np.argmax` retrieves the index of the active class
# (i.e., the class with a label of 1) for each sample in the test dataset.



# Generate and visualize the confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
# This line computes the confusion matrix using scikit-learn's `confusion_matrix` function.
# It compares the true class labels (`y_true_classes`) with the predicted class labels (`y_pred_classes`),
# resulting in a matrix where each element [i, j] indicates the number of instances
# of class i that were predicted as class j.

plt.figure(figsize=(10, 8))
# This line creates a new matplotlib figure with specified dimensions (10 inches wide by 8 inches tall).
# It provides enough space for the matrix and annotations to be clearly visible.

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
# `sns.heatmap` is used here to plot the confusion matrix. 
# - `annot=True` enables annotations inside the squares showing the integer values.
# - `fmt='d'` specifies that these annotations should be formatted as integers.
# - `cmap='Blues'` uses a blue color map to represent different values, providing visual differentiation through color intensity.
# - `xticklabels` and `yticklabels` are set to `class_labels`, which should be a list of class names.
#   These labels make the chart easier to read by replacing numerical class indices with actual class names.

plt.title('Confusion Matrix')
# Sets the title of the plot.

plt.xlabel('Predicted Labels')
# Labels the x-axis as "Predicted Labels".

plt.ylabel('True Labels')
# Labels the y-axis as "True Labels".

plt.savefig('Confusion_Matrix.png')
# Saves the current figure to a file named 'Confusion_Matrix.png'. This is useful for documentation and presentations,
# allowing you to easily share the visual representation of the model's performance.

plt.close()
# Closes the figure, freeing up memory resources. This is particularly important in scripts that create many figures
# or when running in environments with limited memory.


# Generate the classification report
report = classification_report(y_true_classes, y_pred_classes, target_names=class_labels)
# This line generates the classification report which provides key metrics such as precision, recall, and f1-score 
# for each class identified by the classifier. The `target_names` parameter maps the numerical class indices 
# to human-readable names for better clarity in the report output.

print("Classification Report:")
print(report)
# These lines print the classification report to the console, allowing quick viewing of performance metrics 
# for each class directly in the script's output.

# Save the classification report to a text file
with open('Classification_Report.txt', 'w') as f:
    f.write(report)
# This block writes the classification report to a text file named 'Classification_Report.txt'.
# Using `with open()` ensures that the file is properly closed after writing.

# Create a figure and axis to plot the report
fig, ax = plt.subplots(figsize=(6, 4))  # Adjust size as needed
# This line creates a new matplotlib figure and a single subplot. `figsize=(6, 4)` sets the figure size to 6 by 4 inches.

ax.text(0.5, 0.5, report, fontsize=12, ha='center', va='center', transform=ax.transAxes)
# This line places text at the center of the plot. `ha='center'` and `va='center'` set the horizontal and vertical alignment, respectively.
# `fontsize=12` specifies the text size, and `transform=ax.transAxes` ensures that the coordinates (0.5, 0.5) refer to the center of the axes.

ax.axis('off')  # Hide the axes
# This line turns off the axes, which are not needed for text display, making the output cleaner.

# Save the figure containing the text as a PNG file
plt.savefig('Classification_Report.png', dpi=200)  # Use high dpi for better text clarity
# This command saves the current figure to a PNG file with a resolution of 200 dots per inch (dpi),
# which improves the clarity of the text in the image file.

plt.close()  # Close the plot
# This line closes the figure, freeing up memory. This is especially important in scripts that create multiple figures.



print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}, Test Sensitivity: {results[2]}, Test Specificity: {results[3]}")
# Prints out the loss, accuracy, sensitivity, and specificity of the model on the test data, providing a snapshot
# of model performance.

# Save the model in the recommended Keras format
model.save('pet_sound_recognition_model.keras')
# Saves the entire model (architecture, weights, training configuration, state of the optimizer) to a file,
# using the new Keras-specific format which can be useful for later model loading and deployment.

