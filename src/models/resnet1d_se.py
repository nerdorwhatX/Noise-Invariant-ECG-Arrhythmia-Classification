import torch
import torch.nn as nn

import numpy as np
import os

import logging

logger = logging.getLogger(__name__)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        # Initial Convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual Layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(
            ResidualBlock1D(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, 1, 234)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight


class ResNet1DClassifier:
    """
    Object-oriented wrapper for the PyTorch ResNet1D-SE model.
    Provides standard scikit-learn-like fit, predict, save, and load methods.
    Includes Early Stopping and a StepLR learning rate scheduler for optimal training.
    """

    def __init__(self, epochs=100, batch_size=256, lr=0.001, patience=10, device=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = ResNet1D(num_classes=4).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

    def fit(self, X_train, y_train, class_weights=None, validation_split=0.2):
        """
        Trains the ResNet1D model using the provided training data.

        Args:
            X_train (np.ndarray): Training data of shape (N, Length) or (N, 1, Length).
            y_train (np.ndarray): Training labels.
            class_weights (str): If 'balanced', applies balanced class weights to the Loss function.
            validation_split (float): Fraction of training data to use for early stopping validation.
        """
        # Ensure dimensions: (N, Channels, Length)
        if len(X_train.shape) == 2:
            X_train = np.expand_dims(X_train, axis=1)

        # Class weights logic
        if class_weights == "balanced":
            unique_classes = np.unique(y_train)
            weights = compute_class_weight(
                class_weight="balanced", classes=unique_classes, y=y_train
            )
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(
                self.device
            )
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            logger.info(f"Using class weights: {weights}")

        # Split into train/val for Early Stopping
        from sklearn.model_selection import train_test_split

        X_t, X_v, y_t, y_v = train_test_split(
            X_train,
            y_train,
            test_size=validation_split,
            stratify=y_train,
            random_state=42,
        )

        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_t, dtype=torch.float32), torch.tensor(y_t, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_v, dtype=torch.float32), torch.tensor(y_v, dtype=torch.long)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.epochs):
            # Training Phase
            self.model.train()
            total_train_loss, correct_train, total_train = 0, 0, 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()

            train_loss = total_train_loss / total_train
            train_acc = correct_train / total_train

            # Validation Phase
            self.model.eval()
            total_val_loss, correct_val, total_val = 0, 0, 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                    total_val_loss += loss.item() * batch_X.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += batch_y.size(0)
                    correct_val += (predicted == batch_y).sum().item()

            val_loss = total_val_loss / total_val
            val_acc = correct_val / total_val

            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            # Scheduler Step
            self.scheduler.step(val_loss)

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs!")
                    break

        # Restore best weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Restored best model weights based on validation loss.")

    def predict(self, X_test):
        """
        Predicts classes for the given test data.
        """
        if len(X_test.shape) == 2:
            X_test = np.expand_dims(X_test, axis=1)

        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for (batch_X,) in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def save(self, filepath):
        """Saves the model weights to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        """Loads the model weights from disk."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {filepath}")
