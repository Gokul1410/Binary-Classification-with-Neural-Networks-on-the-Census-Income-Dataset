# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset
## Name : Gokul C
## Reg no : 212223240040

## Procedure:
## Step 1: Import Libraries

The program first imports the necessary Python libraries:

PyTorch for building and training the model.

NumPy and Pandas for handling and processing the dataset.

Matplotlib for visualization.

scikit-learn’s shuffle function to randomize dataset rows.

## Step 2: Load and Inspect Dataset

The dataset income.csv is loaded into a Pandas DataFrame.
The program prints:

The total number of rows in the dataset.

The first five rows for preview.

## Step 3: Define Columns

The dataset columns are divided into three types:

Categorical columns (e.g., sex, education, occupation).

Continuous columns (e.g., age, hours-per-week).

Target column (label column).

## Step 4: Convert Categories and Shuffle Data

Each categorical column is converted into category type.
The dataset is shuffled to remove any bias due to ordering.
The index is reset after shuffling.

## Step 5: Create Embedding Sizes

The number of unique categories in each categorical column is calculated.
For each categorical column, an embedding size is created, where:

Input dimension = number of unique categories.

Output dimension = smaller dense representation (limited to max 50).

## Step 6: Convert Data to Tensors

The dataset is converted into PyTorch tensors:

Categorical columns are transformed into numeric codes.

Continuous columns are converted into floating-point tensors.

The target column is flattened into a label tensor.

## Step 7: Split Train and Test

The dataset is divided into:

Training set → the first portion of the data.

Test set → the last portion of the data.

This ensures that the model is trained on one part and tested on another.

## Step 8: Define Model

A custom neural network class TabularModel is created.
It consists of:

Embedding layers for categorical variables.

Dropout for regularization.

Batch Normalization for continuous variables.

Fully connected layers with ReLU activation, BatchNorm, Dropout, and finally a Linear layer producing two outputs (binary classification).

## Step 9: Initialize Model, Loss, and Optimizer

The model is initialized with the embedding sizes, number of continuous features, and output size (2).

The loss function is set to CrossEntropyLoss, since this is a classification problem.

The optimizer chosen is Adam, with a learning rate of 0.001.

## Step 10: Training Loop

The model is trained for 300 epochs. In each epoch:

The model performs a forward pass on the training data.

The loss between predictions and true labels is calculated.

Gradients are reset, backpropagation is performed, and model weights are updated.

The loss is recorded, and printed every 25 epochs.

## Step 11: Plot Training Loss

After training, the program plots a loss curve against epochs.
This shows how the model’s error decreases over time, indicating whether learning is successful.
### Output :

<img width="774" height="568" alt="Screenshot 2025-09-24 090925" src="https://github.com/user-attachments/assets/ba448d0b-1d91-46c8-ad1a-1ca433dca4a6" />
