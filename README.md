Facial Expression Recognition Project
a deep learning project focused on recognizing facial expressions from image data. The project utilizes PyTorch and Weights & Biases for tracking.
The goal of this project is to train a CNN to classify facial expressions into emotion categories. 

Data Preparation:
Dataset Loading
Custom Dataset Class 
The dataset is split into training, validation, and test sets based on the 'Usage' column in data.
Image Transformations: torchvision.transforms are applied for data augmentation.
DataLoaders: PyTorch DataLoaders are set up to create batches of data.

Model Architecture (FacialExpressionCNN):

A custom Convolutional Neural Network is defined using PyTorch's nn.Module.
Batch Normalization and ReLU are used after each layer for training.
Max Pooling layers reduce spatial dimensions after each block.
Dense layers with Dropout layers are used for classification and regularization.

Training and Evaluation:
Training Loop: The model is trained iteratively over a number of epochs, with training and validation steps.
Model Saving: The model's weights are saved when a new best validation accuracy is achieved.
After training, the best-performing model is loaded on the unseen testset.

i tried a CNN model for 3-4 times, it didnt get accuracy score over ~59 in those cases, but i logged the last model of that kind on wandb. than i wrote a new cnn model with enhanced paramenters, trained for 40 epochs which did much better on both validation and test data. the current best test accuracy score is 62.80% as logged on wandb.



Cell 2: Enhanced Facial Expression CNN Model Definition: Defines the neural network.

Cell 3: Training Loop and WandB Integration: Starts the training process.

Results (Expected)
The project aims to achieve a test accuracy of at least 60% for facial expression recognition. The training process and performance metrics can be monitored in real-time on your Weights & Biases dashboard.
