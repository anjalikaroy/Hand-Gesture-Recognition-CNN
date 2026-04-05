# Hand Gesture Recognition using CNN

## Problem Statement
SG Electrics, a smart television manufacturer, wants to enable gesture-based control for its smart TVs. The goal of this project is to build a deep learning model that can correctly recognize hand gestures from short video clips so users can control the TV without a remote.

The five gestures and their corresponding actions are:

- Thumbs Up: Increase volume
- Thumbs Down: Decrease volume
- Stop: Pause the movie
- Left Swipe: Fast forward 10 seconds
- Right Swipe: Fast backward 10 seconds


## Objective
To build an accurate gesture recognition model using deep learning that classifies video sequences into one of five gesture categories.


## Dataset
The dataset consists of gesture video samples, where each sample is a sequence of image frames. Each video belongs to one of five gesture classes.

The notebook uses:
- `train.csv` for training sequence references
- `val.csv` for validation sequence references

Each gesture sample is stored as a folder of image frames.

## Approach
- Read training and validation sequence metadata
- Build a custom data generator to:
  - load image sequences batch-wise
  - sample frames from each video
  - resize and normalize images
  - apply light augmentation using affine transformation
- Experiment with multiple model architectures for video classification
- Compare Conv3D models and CNN + RNN/LSTM models
- Select the best model based on:
  - validation accuracy
  - overfitting behavior
  - number of parameters

## Models Explored
A total of **15 models** were built and evaluated:

### Conv3D Models
- Models 1 to 9
- Different combinations of:
  - number of frames
  - image size
  - kernel sizes
  - dropout values
  - batch size
  - epochs

### CNN + RNN / LSTM Models
- Models 10 to 15
- Used `TimeDistributed Conv2D` layers followed by recurrent modeling
- Tested different frame counts, image sizes, and dropout settings



## Best Model
The final selected model is **Model 5**, a Conv3D architecture.

### Best Model Specifications
- Model Type: Conv3D
- Training Accuracy: 93%
- Validation Accuracy: 87.5%
- Number of Frames: 30
- Image Size: 120 x 120
- Filter Size: 2 x 2 x 2
- Batch Size: 32
- Number of Epochs: 15
- Total Parameters: ~840K


## Final Architecture Summary
The selected model includes:
- Conv3D layers
- Batch Normalization
- ReLU activation
- MaxPooling3D
- Dropout layers
- Flatten layer
- Dense layers
- Softmax output layer for 5-class classification

## Tech Stack
- Python
- NumPy
- OpenCV
- Matplotlib
- TensorFlow / Keras


## Key Insights
- Conv3D models performed better than the CNN + LSTM variants in this project
- Dropout helped reduce overfitting in deeper architectures
- Input frame count, image size, and kernel size had a significant impact on performance
- Model 5 offered the best trade-off between validation accuracy, parameter count, and overfitting

## Results
- Built and compared 15 deep learning models for gesture recognition
- Achieved a final validation accuracy of 87.5% with the selected Conv3D model
- Reduced overfitting through architecture tuning and dropout regularization
- Identified an efficient model suitable for smart TV gesture-based control


## Applications
This project can be used in:
- Smart TV gesture control
- Human-computer interaction
- Touchless interface systems
- Gesture-based automation systems


## Future Improvements
- Use transfer learning with video-based architectures
- Experiment with GRU-based sequence models
- Improve augmentation strategy
- Optimize for real-time inference on edge devices
- Deploy as a real-time webcam-based gesture recognition application


## Project Structure
```text
hand-gesture-recognition-cnn.ipynb
README.md
