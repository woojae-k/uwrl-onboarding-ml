import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def generate_synthetic_data(num_samples=100000):
	X = np.random.rand(num_samples, 3) # Random RGB values
	y = np.array([(1 if r > g + b else 0) for r, g, b in X])
	return X, y

# Generate data and split into train and test sets
X, y = generate_synthetic_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class ColorClassifier(nn.Module):
    def __init__(self):
        super(ColorClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 10)  # 3 input features (R, G, B) and 10 hidden neurons
        self.fc2 = nn.Linear(10, 1)  # 1 output (binary classification)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    

# Instantiate the model
model = ColorClassifier()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

batch_size = 64  # Example batch size
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
#Test the trained model
model.eval()
with torch.no_grad():
    predictions = model(X_test)

    # Convert predictions to binary (round to 0 or 1)
    predictions = (predictions > 0.5).float()
    accuracy = (predictions == y_test).sum() / y_test.size(0)
    print(f"Accuracy: {accuracy:.4f}")