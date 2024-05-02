from models.nlp_model import NLPModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class _PytorchNeuralNet(nn.Module):
    """fully connected neural net with 1 hidden layer, a tanh activation, then an output layer of sz 1 that goes through a sigmoid
    since we are doing binary classification"""

    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.fc1 = nn.Linear(input_sz, hidden_sz)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_sz, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.fc2(self.tanh(self.fc1(x))))
        return out


class FNN(NLPModel):
    """wrapper around pytorch neural net so I can incorporate pyotorch with my pipeline class more easily"""

    def __init__(self, hidden_sz: int):
        self.hidden_sz = hidden_sz

    def train(self, data: np.ndarray, labels: np.ndarray, val_data: np.ndarray = None, val_labels: np.ndarray = None):
        # initialize pytorch neural net
        self.neural_net = _PytorchNeuralNet(data.shape[1], self.hidden_sz)

        # Use Binary Cross Entropy Loss and Stochastic Gradient Descent for training
        loss = nn.BCELoss()
        optimizer = optim.SGD(self.neural_net.parameters(), lr=0.004)

        # Create dataloader for training loop
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        dataset = TensorDataset(data_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=16)

        val_dataloader = None
        if type(val_data) != None and type(val_labels) != None:
            # create dataloader for validation check
            val_dataset = TensorDataset(torch.tensor(
                val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.int32))
            val_dataloader = DataLoader(val_dataset, batch_size=16)

        min_val_accuracy = 0.00
        MAX_EPOCHS = 10
        # training loop
        for epoch in range(MAX_EPOCHS):

            self.neural_net.train()
            for inputs, targets in dataloader:
                # zero out the gradients, get predictions for batch, compute loss from auto-grad, take step in grad's direction
                optimizer.zero_grad()
                outputs = self.neural_net(inputs)
                batch_loss = loss(outputs, targets.unsqueeze(1))
                batch_loss.backward()
                optimizer.step()

            if val_dataloader != None:
                # set model to eval to not compute grads, then get current model's accuracy on validation data
                self.neural_net.eval()
                correct = 0
                total = 0
                for inputs, targets in val_dataloader:
                    outputs = self.neural_net(inputs)
                    predictions = torch.round(outputs).squeeze().long()
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)

                val_acc = round((correct / total) * 100, 2)

                print(
                    f"\tEpoch [{epoch+1}/{MAX_EPOCHS}], Validation Accuracy: {val_acc}")

                # Check for early stopping
                if val_acc < min_val_accuracy + 0.5:
                    print("Validation loss increased. Early stopping.")
                    break
                else:
                    min_val_accuracy = val_acc

        # set model to eval since it is done training
        self.neural_net.eval()

    def eval(self, feature_vector: np.ndarray) -> int:
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)
        result = self.neural_net(feature_tensor)
        bin_class = int(torch.round(result).item())
        return bin_class
