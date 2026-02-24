# Stock-Price-Prediction

## AIM

To develop a **Recurrent Neural Network (RNN)** model for predicting stock prices using historical data.

---

## Problem Statement and Dataset

Stock price prediction is a challenging task due to the non-linear and volatile nature of financial markets. Traditional methods often fail to capture complex temporal dependencies. Deep learning, specifically **Recurrent Neural Networks (RNNs)**, can effectively model time-series dependencies, making them suitable for stock price forecasting.

* **Problem Statement**:
  Build an RNN model to predict the future stock price based on past stock price data.

* **Dataset**:
  A stock market dataset containing **historical daily closing prices** (e.g., Google, Apple, Tesla, or NSE/BSE data).
  The dataset is usually divided into **training and testing sets** after applying normalization and sequence generation.

---

## Design Steps

### Step 1:

Import required libraries such as `torch`, `torch.nn`, `torch.optim`, `numpy`, `pandas`, and `matplotlib`.

### Step 2:

Load the dataset (e.g., stock closing prices from CSV), preprocess it by **normalizing** values between 0 and 1, and create input sequences for training/testing.

### Step 3:

Define the **RNN model architecture** with an input layer, hidden layers, and an output layer to predict stock prices.

### Step 4:

Compile the model using **MSELoss** as the loss function and **Adam optimizer**.

### Step 5:

Train the model on the training data, recording training losses for each epoch.

### Step 6:

Test the trained model on unseen data and visualize results by plotting the **true stock prices vs. predicted stock prices**.

---

## Program

#### Name: SAKTHIVEL S

#### Register Number: 212223220090

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN Layer
        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        
        # Take last time step output
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out
model = RNNModel(input_size=1, hidden_size=50, num_layers=2, num_classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Plot training loss
print('Name:SAKTHIVEL S')
print('Register Number:212223220090')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

```

---

## Output

### True Stock Price, Predicted Stock Price vs Time

<img width="1205" height="724" alt="image" src="https://github.com/user-attachments/assets/f2773faa-6e9d-49cc-aac4-e348accbc986" />




### Predictions

<img width="410" height="95" alt="image" src="https://github.com/user-attachments/assets/e3d9aa62-c011-489f-837a-fbc7c7914e4c" />



---

## Result

The **RNN model** was successfully implemented for **stock price prediction**.
