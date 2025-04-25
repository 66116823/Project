import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import optuna
from tqdm import tqdm
import numpy as np
import random
from typing import List, Optional

from cnn_preprocessing import (
    load_video,
    # select_frames,
    split_dataset,
    folder_paths,
    load_videos_from_folders
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4

class HandGestureDataset(Dataset):
    def __init__(self,
                 video_paths: List[str],
                 labels: List[int],
                 sequence_length: int = 15,
                 num_sequences: int = 1,
                 transform: Optional[object] = None):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.transform = transform

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        try:
            frames = load_video(self.video_paths[idx])
            if frames.size == 0:
                # Return a zero tensor with the correct shape
                return (
                    torch.zeros((self.num_sequences, self.sequence_length, 3, 256, 192)),
                    torch.tensor(self.labels[idx], dtype=torch.long)
                )

            # Ensure we have enough frames
            if len(frames) < self.sequence_length:
                # Pad with duplicated frames if necessary
                last_frame = frames[-1]
                num_padding = self.sequence_length - len(frames)
                padding_frames = np.tile(last_frame, (num_padding, 1, 1, 1))
                frames = np.concatenate([frames, padding_frames])
            elif len(frames) > self.sequence_length:
                # Take evenly spaced frames
                indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
                frames = frames[indices]

            # Create sequences
            sequences = []
            for _ in range(self.num_sequences):
                sequence_frames = []
                for frame in frames:
                    if self.transform:
                        frame = self.transform(frame)
                    else:
                        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
                    sequence_frames.append(frame)
                sequences.append(torch.stack(sequence_frames))

            # Stack all sequences
            sequences_tensor = torch.stack(sequences)
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

            return sequences_tensor, label_tensor

        except Exception as e:
            print(f"Error processing video {self.video_paths[idx]}: {e}")
            return (
                torch.zeros((self.num_sequences, self.sequence_length, 3, 256, 192)),
                torch.tensor(self.labels[idx], dtype=torch.long)
            )



class CNN(nn.Module):
    def __init__(self, activation, dropout):
        super(CNN, self).__init__()
        act_fn = {
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }[activation]

        self.features = nn.Sequential(
            # Input size: 256*192*3
            # Spatial extend of each one (kernelConv size), F = 7
            # Slide size (strideConv), S = 2
            # Padding, P = 2
            ## Width: ((256 - 7 + 2 * 2) / 2) + 1 = 127.5 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((192 - 7 + 2 * 2) / 2) + 1 =  95.5 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 16
            ## Output Conv Layer1: 127.5  * 95.5 * 16

            nn.Conv2d(3, 16, kernel_size=7, stride= 2, padding=2),
            act_fn,
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm

            # Max Pooling Layer 1
            # Input size: 127.5  * 95.5 * 16
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 1
            ## Width: ((127.5 - 2) / 2) + 1 = 63.75  #*# (( W2 - F ) / S ) + 1
            ## High: ((95.5 - 2) / 2) + 1 = 47.75 #*# (( H2 - F ) / S ) + 1
            ## Depth: 16
            ### Output Max Pooling Layer 1: 63.75 * 47.75 * 16
            nn.MaxPool2d(2, 2),

            # Conv Layer 2
            # Input size:  63.75 * 47.75 * 16
            # Spatial extend of each one (kernelConv size), F = 7
            # Slide size (strideConv), S = 2
            # Padding, P = 2
            ## Width: ((63.75 - 7 + 2 * 2) / 2) + 1 =  31.375 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((47.75  - 7 + 2 * 2) / 2) + 1 =  23.372  #*# W2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 32
            ## Output Conv Layer1: 31.375 * 23.372 * 32
            nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=2),
            act_fn,
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm

            # Max Pooling Layer 2
            # Input size: 31.375 * 23.372 * 32
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 2
            ## Width: ( 31.375 - 2) / 2) + 1 = 15.6875
            ## High: (( 23.372 - 2) / 2) + 1 = 11.686
            ## Depth: 32
            ### Output Max Pooling Layer 2:15.6875 * 11.686 * 32
            nn.MaxPool2d(2, 2),

            # Conv Layer 3
            # Input size: 15.6875 * 11.686 * 32
            # Spatial extend of each one (kernelConv size), F = 7
            # Slide size (strideConv), S = 2
            # Padding, P = 2
            ## Width: (( 15.6875  - 7+ 2 * 2) / 2) + 1 = 7.34375  #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((11.686 - 7+ 2 * 2) / 2) + 1 = 5.343 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 64
            ## Output Conv Layer1: 7.34375  * 5.343 * 64
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=2),
            act_fn,
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm

            # Max Pooling Layer 3
            # Input size: 7.34375  * 5.343 * 64
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 2
            ## Width: ((7.34375 - 2) / 2) + 1 = 3.67
            ## High: ((5.343 - 2) / 2) + 1 = 2.67
            ## Depth: 64
            ### Output Max Pooling Layer 2: 3.67 * 2.67 * 64
            nn.MaxPool2d(2, 2),

            ##### ______________________ ถึง 3 ชั้น เพราะเลขน้อยมากแล้วว ____________________________

            # # Conv Layer 4
            # # Input size: 3.67 * 2.67 * 64
            # # Spatial extend of each one (kernelConv size), F = 7
            # # Slide size (strideConv), S = 2
            # # Padding, P = 2
            # ## Width: ((3.67 - 7 + 2 * 2) / 2) + 1 =   #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            # ## High: ((2.67  - 7 + 2 * 2) / 2) + 1 = 24 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            # ## Depth: 128
            # ## Output Conv Layer1: 32 * 24 * 128
            # nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            # act_fn,
            # nn.BatchNorm2d(128),
           # nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm

            #
            # # Max Pooling Layer 4
            # # Input size: 32 * 24 * 128
            # ## Spatial extend of each one (kernelMaxPool size), F = 2
            # ## Slide size (strideMaxPool), S = 2
            # # Output Max Pooling Layer 2
            # ## Width: ((32 - 2) / 2) + 1 = 16
            # ## High: ((24 - 2) / 2) + 1 = 12
            # ## Depth: 128
            # ### Output Max Pooling Layer 2: 16 * 12 * 128
            # nn.MaxPool2d(2, 2),

            # # Conv Layer 5
            # # Input size: 16 * 12 * 128
            # # Spatial extend of each one (kernelConv size), F = 5
            # # Slide size (strideConv), S = 1
            # # Padding, P = 2
            # ## Width: ((16 - 5 + 2 * 2) / 1) + 1 =  16 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            # ## High: ((12  - 5 + 2 * 2) / 1) + 1 = 12 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            # ## Depth: 256
            # ## Output Conv Layer1: 16 * 12 * 256
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            # act_fn,
            # nn.BatchNorm2d(256),
            # nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm

            #
            # # Max Pooling Layer 5
            # # Input size: 16 * 12 * 256
            # ## Spatial extend of each one (kernelMaxPool size), F = 2
            # ## Slide size (strideMaxPool), S = 2
            # # Output Max Pooling Layer 2
            # ## Width: ((16 - 2) / 2) + 1 = 8
            # ## High: ((12 - 2) / 2) + 1 = 6
            # ## Depth: 256
            # ### Output Max Pooling Layer 2: 8* 6* 256
            # nn.MaxPool2d(2, 2),

        )

    def forward(self, x):
        return self.features(x)

# input_size = 63 * 47 * 16 # for 1 layers CNN
# input_size = 15 * 11 * 32 # for 2 layers CNN
input_size = 3 * 2 * 64 # for 3 layers CNN


class CNN_LSTM(nn.Module):
    def __init__(self, hidden_units, hidden_layers, activation, dropout):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN(
            activation=activation,
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


def objective(trial):
    # Print current trial number
    print(f"\n{'=' * 50}")
    print(f"Trial #{trial.number}")
    print(f"{'=' * 50}")

    # Hyperparameter optimization
    batch_size = trial.suggest_int('batch_size', 8, 256)
    activation = trial.suggest_categorical('activation', ['ReLU', 'Sigmoid', 'tanh'])
    hidden_units = trial.suggest_int('hidden_units', 50, 150)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 10)
    num_epochs = trial.suggest_int('num_epochs', 200, 1000)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.099, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.7, log=True)


    # Print trial parameters
    print("\nTrial Parameters:")
    print(f"-- Batch Size: {batch_size}")
    print(f"-- Activation Function: {activation}")
    print(f"-- Hidden Units: {hidden_units}")
    print(f"-- Hidden Layers: {hidden_layers}")
    print(f"-- Number of Epochs: {num_epochs}")
    print(f"-- Learning Rate: {learning_rate:.6f}")
    print(f"-- Dropout: {dropout:.6f}")


    print(f"{'=' * 50}")



    # [Code for data loading and model setup remains the same]
    video_paths, labels = load_videos_from_folders(folder_paths)
    train_paths, test_paths, train_labels, test_labels = split_dataset(
        video_paths, labels, test_ratio=0.2)


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_sequences = 1
    train_dataset = HandGestureDataset(train_paths, train_labels, num_sequences=num_sequences, transform=transform)
    test_dataset = HandGestureDataset(test_paths, test_labels, num_sequences=num_sequences, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CNN_LSTM(
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # # Early stopping parameters
    # patience = 5
    # early_stopping_counter = 0

    # Training loop
    best_val_acc = 0.0
    print("\nTraining Progress:")
    print("-" * 50)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)
        # print(f"Epoch {epoch+1}")
        print(f"\nTraining Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print("-" * 30)

        # # Check for early stopping
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     early_stopping_counter = 0  # Reset counter when improvement is seen
        # else:
        #     early_stopping_counter += 1  # No improvement, increase counter
        #
        # # If no improvement for 'patience' epochs, stop training
        # if early_stopping_counter >= patience:
        #     print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy.")
        #     break



    # Print trial summary
    print(f"\nTrial #{trial.number} Summary:")
    print(f"{'=' * 30}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    # print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Parameters:")
    print(f"-- Batch Size: {batch_size}")
    print(f"-- Activation Function: {activation}")
    print(f"-- Hidden Units: {hidden_units}")
    print(f"-- Hidden Layers: {hidden_layers}")
    print(f"-- Number of Epochs: {num_epochs}")
    print(f"-- Learning Rate: {learning_rate:.6f}")
    print(f"-- Dropout: {dropout:.6f}")
    print(f"{'=' * 50}")

    # Store results
    # trial.set_user_attr('test_accuracy', test_acc)
    trial.set_user_attr('best_val_accuracy', best_val_acc)

    return best_val_acc


def main():
    # GPU information
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Optuna setup
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, catch=(Exception,))

    print("\nBest Trial Results:")
    print("=" * 50)
    trial = study.best_trial
    # Find which trial number was the best
    best_trial_number = None
    for trial_idx, t in enumerate(study.trials):
        if t.number == trial.number:
            best_trial_number = trial_idx + 1
            break

    print(f"Best Trial Number: {best_trial_number} (Trial #{trial.number})")
    print(f"Best Accuracy: {trial.value:.2f}%")
    print("\nBest Parameters:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
