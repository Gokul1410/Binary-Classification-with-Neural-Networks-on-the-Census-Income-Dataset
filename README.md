# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset
## Name : Gokul C
## Reg no : 212223240040

### Program:

```
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df = pd.read_csv('income.csv')

# Check the length and preview of the dataframe
print(f"Length of the dataset: {len(df)}")
print("First 5 rows of the dataset:")
print(df.head())

# Define categorical and continuous columns
cat_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']

# Print out column counts
print(f'cat_cols has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col has {len(y_col)} column')

# Convert categorical columns to category type
for col in cat_cols:
    df[col] = df[col].astype('category')

# Shuffle the data
df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)

# Preview the first 5 rows of the shuffled dataset
print(df.head())

# Create embedding sizes for each categorical column
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(f"Embedding sizes: {emb_szs}")

# Convert categorical data to numeric codes
cats = torch.tensor(np.stack([df[col].cat.codes.values for col in cat_cols], axis=1), dtype=torch.long)

# Convert continuous data to tensors
conts = torch.tensor(np.stack([df[col].values for col in cont_cols], axis=1), dtype=torch.float32)

# Convert labels to tensor
y = torch.tensor(df[y_col].values).flatten()

# Split data into training and test sets
b = 30000  # batch size
t = 5000   # test size

cat_train = cats[:b-t]
cat_test  = cats[b-t:b]
con_train = conts[:b-t]
con_test  = conts[b-t:b]
y_train   = y[:b-t]
y_test    = y[b-t:b]

print(f"cat_train shape: {cat_train.shape}")
print(f"con_train shape: {con_train.shape}")
print(f"y_train shape: {y_train.shape}")

class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        
        # Create embedding layers for categorical columns
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        
        # Dropout for embeddings and batch normalization for continuous columns
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # Build hidden layers
        layerlist = []
        n_emb = sum((nf for ni, nf in emb_szs))
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        # Process categorical data through embedding layers
        embeddings = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        # Process continuous data through batch normalization
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)

        # Pass through the hidden layers
        x = self.layers(x)
        return x
# Initialize the model, loss function, and optimizer
torch.manual_seed(33)
model = TabularModel(emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)

# Use CrossEntropyLoss for classification and Adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)# Training loop
epochs = 300
losses = []

for i in range(1, epochs + 1):
    # Forward pass
    y_pred = model(cat_train, con_train)
    
    # Compute loss
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    # Print loss every 25 epochs
    if i % 25 == 1:
        print(f'Epoch {i:3} Loss: {loss.item():.8f}')
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Curve')
plt.show()

```

### Output :

<img width="774" height="568" alt="image" src="https://github.com/user-attachments/assets/5839dcd4-6824-4ae7-88e4-0c8f19af4b9e" />
