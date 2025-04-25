import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional
import gc
import os
import matplotlib.pyplot as plt
from typing import List, Dict
import time
import time
import psutil
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns

from cnn_preprocessing import VideoPreprocessor
preprocessor = VideoPreprocessor()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4

# Create directory for saving picture
plot = r"D:\cnn_lstm\Plot"
os.makedirs(plot, exist_ok=True)

class TrainingVisualizer:
    def __init__(self, save_dir: str, dropout: float):
        """
        Initialize the training visualizer.

        Args:
            save_dir: Directory to save the generated plots
            dropout: Dropout value used in training
        """
        self.save_dir = save_dir
        self.dropout = dropout
        os.makedirs(save_dir, exist_ok=True)

        # Store metrics during training
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'gpu_memory': [],
            'batch_times': [],
            'epoch_times': []
        }

        # Use default style instead of seaborn
        plt.style.use('default')

    def update_metrics(self, epoch_metrics: Dict):
        """Update stored metrics with new values"""
        for key, value in epoch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def plot_all_metrics(self):
        """Generate individual plots for each metric"""
        # Ensure we have data to plot
        if not self.metrics['train_loss']:
            print("No metrics to plot.")
            return

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        plt.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f'loss_dropout_{self.dropout}.png'))
        plt.close()

        # Plot training and validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, self.metrics['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f'accuracy_dropout_{self.dropout}.png'))
        plt.close()

        # Plot GPU memory usage
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['gpu_memory'], 'g-', linewidth=2)
        plt.title('GPU Memory Usage Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Memory Usage (GB)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f'gpu_memory_dropout_{self.dropout}.png'))
        plt.close()

        # Plot epoch times
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['epoch_times'], 'b-', linewidth=2)
        plt.title('Epoch Training Times')
        plt.xlabel('Epochs')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f'epoch_times_dropout_{self.dropout}.png'))
        plt.close()

        # Plot average batch times
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['batch_times'], 'r-', linewidth=2)
        plt.title('Average Batch Processing Times')
        plt.xlabel('Epochs')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f'batch_times_dropout_{self.dropout}.png'))
        plt.close()


class HandGestureDataset(Dataset):
    def __init__(self,
                 sequences: np.ndarray,
                 labels: np.ndarray,
                 num_sequences: int = 1,
                 transform: Optional[object] = None):
        self.sequences = sequences
        self.labels = labels
        self.num_sequences = num_sequences
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        try:
            # Get the sequence and convert to tensor
            sequence = self.sequences[idx]

            # Apply transform if specified
            if self.transform:
                transformed_frames = []
                for frame in sequence:
                    transformed_frames.append(self.transform(frame))
                sequence = np.stack(transformed_frames)

            # Convert to tensor and normalize
            sequence_tensor = torch.from_numpy(sequence.transpose(0, 3, 1, 2)).float() / 255.0

            # Reshape for multiple sequences if needed
            sequences_tensor = sequence_tensor.unsqueeze(0).repeat(self.num_sequences, 1, 1, 1, 1)

            # Get label
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

            return sequences_tensor, label_tensor

        except Exception as e:
            print(f"Error processing sequence {idx}: {e}")
            return (
                torch.zeros((self.num_sequences, 15, 3, 256, 192)),
                torch.tensor(self.labels[idx], dtype=torch.long)
            )


class CNN(nn.Module):
    def __init__(self, dropout):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # Input size: 256*192*3
            # Spatial extend of each one (kernelConv size), F = 5
            # Slide size (strideConv), S = 2
            # Padding, P = 2
            ## Width: ((256 - 5 + 2 * 2) / 2) + 1 = 128.5 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((192 - 5 + 2 * 2) / 2) + 1 =  96.5 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 16
            ## Output Conv Layer1:  128.5 * 96.5 * 16
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm

            # Max Pooling Layer 1
            # Input size: 128.5 * 96.5 * 16
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 1
            ## Width: ((128.5 - 2) / 2) + 1 = 64.25 #*# (( W2 - F ) / S ) + 1
            ## High: ((96.5 - 2) / 2) + 1 = 48.25 #*# (( H2 - F ) / S ) + 1
            ## Depth: 16
            ### Output Max Pooling Layer 1: 64.25 * 48.25 * 16
            nn.MaxPool2d(2, 2),
            #
            # # Conv Layer 2
            # # Input size:  64.25 * 48.25 * 16
            # # Spatial extend of each one (kernelConv size), F = 5
            # # Slide size (strideConv), S = 2
            # # Padding, P = 2
            # ## Width: ((64.25 - 5 + 2 * 2) / 2) + 1 =  32.625 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            # ## High: ((48.25 - 5 + 2 * 2) / 2) + 1 =   24.625 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            # ## Depth: 32
            # ## Output Conv Layer1: 32.625 * 24.625 * 32
            # nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
            # nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm
            #
            # # Max Pooling Layer 2
            # # Input size: 32.625 * 24.625 * 32
            # ## Spatial extend of each one (kernelMaxPool size), F = 2
            # ## Slide size (strideMaxPool), S = 2
            # # Output Max Pooling Layer 2
            # ## Width: ((32.625 - 2) / 2) + 1 = 16.312
            # ## High: ((24.625 - 2) / 2) + 1 = 12.312
            # ## Depth: 32
            # ### Output Max Pooling Layer 2: 16.312 * 12.312 * 32
            # nn.MaxPool2d(2, 2),
            #
            # # Conv Layer 3
            # # Input size: 16.312 * 12.312 * 32
            # # Spatial extend of each one (kernelConv size), F = 5
            # # Slide size (strideConv), S = 1
            # # Padding, P = 2
            # ## Width: ((16.312 - 5 + 2 * 2) / 2) + 1 =  8.656 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            # ## High: ((12.312 - 5 + 2 * 2) / 2) + 1 =  6.656  #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            # ## Depth: 64
            # ## Output Conv Layer1: 8.656 * 6.656 * 64
            # nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            # nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm

            # Max Pooling Layer 3
            # Input size: 8.656 * 6.656 * 64
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 2
            ## Width: ((8.656 - 2) / 2) + 1 = 4.328
            ## High: ((6.656 - 2) / 2) + 1 = 3.328
            ## Depth: 64
            ### Output Max Pooling Layer 2: 4.328 * 3.328 * 64
            # nn.MaxPool2d(2, 2),
            #
            # # Conv Layer 4
            # # Input size: 4.328 * 3.328 * 64
            # # Spatial extend of each one (kernelConv size), F = 5
            # # Slide size (strideConv), S = 2
            # # Padding, P = 2
            # ## Width: ((4.328 - 5 + 2 * 2) / 2) + 1 =  2.66 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            # ## High: ((3.328  - 5 + 2 * 2) / 2) + 1 = 2.16 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            # ## Depth: 128
            # ## Output Conv Layer1: 2.66 * 2.16 * 128
            # nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),
            # nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm
            #
            # # Max Pooling Layer 4
            # # Input size: 2.66 * 2.16 * 128
            # ## Spatial extend of each one (kernelMaxPool size), F = 2
            # ## Slide size (strideMaxPool), S = 2
            # # Output Max Pooling Layer 2
            # ## Width: ((2.66 - 2) / 2) + 1 = 1.33
            # ## High: ((2.16 - 2) / 2) + 1 = 1.08
            # ## Depth: 128
            # ### Output Max Pooling Layer 2: 1.33 * 1.08 * 128
            # nn.MaxPool2d(2, 2),
            #
            # # Conv Layer 5
            # # Input size: 1.33 * 1.08 * 128
            # # Spatial extend of each one (kernelConv size), F = 5
            # # Slide size (strideConv), S = 2
            # # Padding, P = 2
            # ## Width: ((1.33 - 5 + 2 * 2) / 2) + 1 = 1.165  #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            # ## High: ((1.08 - 5 + 2 * 2) / 2) + 1 = 1.04 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            # ## Depth: 256
            # ## Output Conv Layer1: 1.165 * 1.04 * 256
            # nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
            # nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm
            #
            # # Max Pooling Layer 5
            # # Input size: 1.165 * 1.04 * 256
            # ## Spatial extend of each one (kernelMaxPool size), F = 2
            # ## Slide size (strideMaxPool), S = 2
            # # Output Max Pooling Layer 2
            # ## Width: ((1.165 - 2) / 2) + 1 = 0.58
            # ## High: ((1.04 - 2) / 2) + 1 = 0.52
            # ## Depth: 256
            # ### Output Max Pooling Layer 2: 0.58* 0.52* 256
            # nn.MaxPool2d(2, 2),

        )

    def forward(self, x):
        return self.features(x)

input_size = 64 * 48 * 16 # for 1 layers CNN
# input_size = 16 * 12 * 32 # for 2 layers CNN
# input_size = 4 * 3 * 64 # for 3 layers CNN
# input_size = 1 * 1 * 128 # for 4 layers CNN
# input_size = 0 * 0 * 256 # for 5 layers CNN

class CNN_LSTM(nn.Module):
    def __init__(self, hidden_units, hidden_layers, dropout):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN(
            dropout=dropout
        )

        self.lstm = nn.LSTM(
            input_size=input_size ,
            hidden_size=hidden_units,
            num_layers=hidden_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_units, num_classes) ## numclasses คือจำนวน label ที่จะให้ทำนายตอนท้ายสุด

    def forward(self, x):
        # x shape: (batch, num_sequences, seconds, channels, height, width)
        batch_size = x.size(0)
        num_sequences = x.size(1)
        sequence_length = x.size(2)

        # Reshape for CNN
        x = x.view(batch_size * num_sequences * sequence_length, *x.size()[3:])
        x = self.cnn(x)

        # Reshape for LSTM
        x = x.view(batch_size * num_sequences, sequence_length, -1)
        lstm_out, _ = self.lstm(x)

        # Take final output
        # x = self.fc(lstm_out[:, -1, :])
        x = self.fc(self.dropout(lstm_out[:, -1, :]))  # เพิ่ม dropout ก่อน fully connected layer

        # Reshape back
        x = x.view(batch_size, num_sequences, -1)

        # Average predictions
        x = x.mean(dim=1)

        return x


class BenchmarkMetrics:
    def __init__(self):
        self.batch_times = []
        self.forward_times = []
        self.backward_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []

    def calculate_averages(self):
        return {
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'avg_forward_time': np.mean(self.forward_times) if self.forward_times else 0,
            'avg_backward_time': np.mean(self.backward_times) if self.backward_times else 0,
            'avg_gpu_memory': np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            'avg_cpu_memory': np.mean(self.cpu_memory_usage) if self.cpu_memory_usage else 0
        }

    def print_summary(self):
        averages = self.calculate_averages()
        print("\nBenchmark Summary:")
        print("=" * 50)
        print(f"Average Batch Processing Time: {averages['avg_batch_time']:.4f} seconds")
        print(f"Average Forward Pass Time: {averages['avg_forward_time']:.4f} seconds")
        print(f"Average Backward Pass Time: {averages['avg_backward_time']:.4f} seconds")
        print(f"Average GPU Memory Usage: {averages['avg_gpu_memory']:.2f} GB")
        print(f"Average CPU Memory Usage: {averages['avg_cpu_memory']:.2f} GB")
        print("=" * 50)


def train_model():
    print(f"\n{'=' * 50}")
    print(f"CNN-LSTM Training")
    print(f"{'=' * 50}")

    # Fixed hyperparameters
    num_epochs = 200
    dropout = 0.1
    learning_rate = 0.001
    batch_size = 37
    hidden_units = 150
    hidden_layers = 3

    # Initialize Visualizer and Benchmark Metrics
    visualizer = TrainingVisualizer(save_dir=plot, dropout=dropout)
    benchmark = BenchmarkMetrics()

    # Print parameters
    print("\nParameters:")
    print(f"-- Batch Size: {batch_size}")
    print(f"-- Hidden Units: {hidden_units}")
    print(f"-- Hidden Layers: {hidden_layers}")
    print(f"-- Number of Epochs: {num_epochs}")
    print(f"-- Learning Rate: {learning_rate:.3f}")
    print(f"-- Dropout: {dropout:.1f}")

    print(f"{'=' * 50}")

    # Get train, validation and test data
    X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.get_train_test_data()

    # Combine training and validation sets
    train_data = np.concatenate([X_train, X_val])
    train_labels = np.concatenate([y_train, y_val])

    num_sequences = 1
    train_dataset = HandGestureDataset(train_data, train_labels, num_sequences=num_sequences)
    test_dataset = HandGestureDataset(X_test, y_test, num_sequences=num_sequences)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN_LSTM(
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    print("\nTraining Progress:")
    print("-" * 50)

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_losses = []
        epoch_start_time = time.time()
        batch_times_epoch = []
        gpu_memory_epoch = []

        # Training phase with benchmarking
        for batch_idx, (sequences, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):
            batch_start_time = time.time()

            sequences, labels = sequences.to(device), labels.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass timing
            forward_start = time.time()
            outputs = model(sequences)
            forward_time = time.time() - forward_start
            benchmark.forward_times.append(forward_time)

            # Loss calculation and backward pass timing
            loss = criterion(outputs, labels)
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start
            benchmark.backward_times.append(backward_time)

            # Store metrics
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Batch timing
            batch_time = time.time() - batch_start_time
            batch_times_epoch.append(batch_time)
            benchmark.batch_times.append(batch_time)

            # Memory tracking
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9  # Convert to GB
                gpu_memory_epoch.append(gpu_memory)
                benchmark.gpu_memory_usage.append(gpu_memory)

            # CPU Memory tracking
            process = psutil.Process()
            cpu_memory = process.memory_info().rss / 1e9  # Convert to GB
            benchmark.cpu_memory_usage.append(cpu_memory)

            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        train_acc = 100 * train_correct / train_total
        train_loss = np.mean(train_losses)
        epoch_time = time.time() - epoch_start_time

        # Validation phase with metrics
        model.eval()
        val_correct = 0
        val_total = 0
        val_losses = []
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)

                forward_start = time.time()
                outputs = model(sequences)
                forward_time = time.time() - forward_start
                benchmark.forward_times.append(forward_time)

                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        val_acc = 100 * val_correct / val_total
        val_loss = np.mean(val_losses)
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
        val_precision = precision_score(val_true_labels, val_predictions, average='weighted')
        val_recall = recall_score(val_true_labels, val_predictions, average='weighted')

        # Update visualization metrics
        epoch_metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gpu_memory': np.mean(gpu_memory_epoch) if gpu_memory_epoch else 0,
            'batch_times': np.mean(batch_times_epoch),
            'epoch_times': epoch_time
        }
        visualizer.update_metrics(epoch_metrics)

        # Save plots after each epoch
        visualizer.plot_all_metrics()

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Print epoch results with benchmarking info
        print(f"\nEpoch {epoch + 1}")
        print(f"Training Metrics:")
        print(f"- Accuracy: {train_acc:.2f}%")
        print(f"- Loss: {train_loss:.4f}")
        print(f"Validation Metrics:")
        print(f"- Accuracy: {val_acc:.2f}%")
        print(f"- F1 Score: {val_f1:.4f}")
        print(f"- Precision: {val_precision:.4f}")
        print(f"- Recall: {val_recall:.4f}")

        # Print current batch benchmarks
        batch_averages = benchmark.calculate_averages()
        print("\nCurrent Benchmarks:")
        print(f"- Avg Batch Time: {batch_averages['avg_batch_time']:.4f} seconds")
        print(f"- Avg Forward Time: {batch_averages['avg_forward_time']:.4f} seconds")
        print(f"- Avg Backward Time: {batch_averages['avg_backward_time']:.4f} seconds")
        print(f"- Current GPU Memory: {batch_averages['avg_gpu_memory']:.2f} GB")
        print("-" * 30)

    # Print final benchmark summary
    benchmark.print_summary()

    # Final evaluation and summary
    print(f"\nTraining Summary:")
    print(f"{'=' * 30}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Validation Metrics:")
    print(f"- F1 Score: {val_f1:.4f}")
    print(f"- Precision: {val_precision:.4f}")
    print(f"- Recall: {val_recall:.4f}")
    print(f"Parameters:")
    print(f"-- Batch Size: {batch_size}")
    print(f"-- Hidden Units: {hidden_units}")
    print(f"-- Hidden Layers: {hidden_layers}")
    print(f"-- Number of Epochs: {num_epochs}")
    print(f"-- Learning Rate: {learning_rate:.6f}")
    print(f"-- Dropout: {dropout:.6f}")
    print(f"{'=' * 50}")

    return model, best_val_acc, test_loader


def evaluate_final_results(model, test_loader, device):
    print(f"\n{'=' * 50}")
    print(f"Final Model Evaluation")
    print(f"{'=' * 50}")

    model.eval()
    test_pred = []
    test_true = []

    # Generate predictions
    print("\nGenerating predictions for test set...")
    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc="Evaluating"):
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions = torch.argmax(outputs, dim=1)
            test_pred.extend(predictions.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(test_true, test_pred) * 100
    f1 = f1_score(test_true, test_pred, average='weighted')
    precision = precision_score(test_true, test_pred, average='weighted')
    recall = recall_score(test_true, test_pred, average='weighted')

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 50)
    class_names = ["No_Requests", "Person_1_Requests",
                   "Person_2_Requests", "Person_1_and_Person_2_Requests"]
    cm = confusion_matrix(test_true, test_pred)

    print("\nClass-wise Results:")
    print("-" * 50)
    print(classification_report(test_true, test_pred, target_names=class_names))

    # Print final metrics
    print("\nFINAL TEST SET RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print("=" * 50)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    # Save confusion matrix in the specified plot directory (D:\cnn_lstm\Plot)
    confusion_matrix_path = os.path.join(plot, f'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"\nConfusion matrix saved to: {confusion_matrix_path}")


def main():
    # GPU information
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Train model
    model, best_acc, test_loader = train_model()
    print(f"Training completed with best accuracy: {best_acc:.2f}%")

    # Evaluate final results
    evaluate_final_results(model, test_loader, device)


if __name__ == "__main__":
    main()
