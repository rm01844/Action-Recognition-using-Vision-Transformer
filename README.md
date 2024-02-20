# Action-Recognition-using-Vision-Transformer
This repository contains the implementation of an action recognition system using Vision Transformers, specifically leveraging the facebook/timesformer-base-finetuned-k400 model. The system is designed to work on the HMDB dataset, a collection of video clips from various sources aimed at human motion recognition.

# Prerequisites

Python 3.8 or higher
PyTorch 1.8 or higher
torchvision 0.9 or higher
transformers 4.5 or higher
PIL (Python Imaging Library)
NumPy
scikit-learn

# Installation
Clone the repository to your local machine:
git clone https://github.com/your-github-username/action-recognition-vision-transformers.git
cd action-recognition-vision-transformers

Install the required dependencies:
pip install torch torchvision transformers Pillow numpy scikit-learn

# Dataset Preparation
Download the HMDB dataset and place it in a directory accessible to the project.
Update the dataset_zip_path and extract_folder variables in the script to point to your dataset zip file and the desired extraction directory, respectively.

# Usage
To run the action recognition system, execute the following command:
python action_recognition.py
This will start the process of dataset preparation, model training, and evaluation. The script will output the training and validation accuracy for each epoch, and finally, the top-5 accuracy on the validation set.

# Model Training
The model is trained using the Vision Transformer architecture pre-trained on the Kinetics-400 dataset. The training procedure includes:

Extracting frames from video clips and applying transformations.
Splitting the dataset into training and validation sets.
Defining the model, loss function, and optimizer.
Training the model for a specified number of epochs.
Evaluating the model on the validation set.

# Evaluation
After training, the model is evaluated using the validation set to calculate the top-1 and top-5 accuracies. These metrics provide insight into how well the model performs at recognizing actions from a set of predefined categories.

# Saving and Loading Models
The trained model is saved to disk and can be loaded for further evaluation or inference on new video data.
