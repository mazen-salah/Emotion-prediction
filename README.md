# Facial Expression Recognition

This Jupyter Notebook contains the code for a Facial Expression Recognition system. The system is designed to recognize facial expressions in images and real-time video streams.

## Table of Contents
- [Introduction](#introduction)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Saving the Model](#saving-the-model)
- [Real-time Facial Expression Recognition](#real-time-facial-expression-recognition)

## Introduction

The goal of this project is to build a Facial Expression Recognition system using deep learning. This system can recognize seven different facial expressions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. It uses a Convolutional Neural Network (CNN) to classify facial expressions in images and can also perform real-time facial expression recognition using a webcam.

## Data Exploration

The first step in this project involves exploring the dataset. The code provided checks the number of images available for each expression in the training dataset.

## Data Preprocessing

Before training the model, the data needs to be preprocessed. The code provided initializes data generators for training and validation. These generators perform data augmentation and prepare the data for the model.

## Model Architecture

The model architecture is defined in this section. It's a CNN with several convolutional layers, batch normalization, activation functions, and dropout layers. The model is designed to classify facial expressions into seven categories.

## Training the Model

The model is trained with the training data and validated using the validation data. The training process includes defining callbacks like learning rate reduction and model checkpointing. The history of the training process is also recorded.

## Saving the Model

After training, the model's architecture and weights are saved to files. This allows the model to be loaded and used for inference without retraining.

## Real-time Facial Expression Recognition

The notebook includes code for real-time facial expression recognition using a webcam. It captures video frames, detects faces, and predicts the facial expressions in real-time. The results are displayed on the video stream.

This README provides an overview of the Facial Expression Recognition project. For detailed code and explanations, please refer to the notebook.
