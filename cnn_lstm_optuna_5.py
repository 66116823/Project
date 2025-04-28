import torch  # นำเข้า PyTorch สำหรับการสร้างโมเดลและการคำนวณ
import torch.nn as nn  # นำเข้าโมดูล neural network จาก PyTorch
from torch.utils.data import Dataset, DataLoader  # นำเข้า Dataset และ DataLoader สำหรับการจัดการข้อมูล
import optuna  # นำเข้า Optuna สำหรับการทำ hyperparameter optimization
from tqdm import tqdm  # นำเข้า tqdm สำหรับการแสดง progress bar
import numpy as np  # นำเข้า NumPy สำหรับการคำนวณเชิงตัวเลข
from typing import List, Optional  # นำเข้า typing สำหรับการกำหนดประเภทข้อมูล
import gc  # นำเข้า gc สำหรับการจัดการหน่วยความจำ
import os  # นำเข้า os สำหรับการจัดการไฟล์และโฟลเดอร์
from collections import defaultdict  # นำเข้า defaultdict สำหรับเก็บข้อมูล
import matplotlib.pyplot as plt  # นำเข้า Matplotlib สำหรับการสร้างกราฟ

# นำเข้าสำหรับการประมวลผลวิดีโอ
from cnn_preprocessing import ( 
    load_video,
    split_dataset,
    folder_paths,
    load_videos_from_folders
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # กำหนดอุปกรณ์สำหรับการประมวลผล (GPU หรือ CPU)
num_classes = 4  # จำนวนคลาสที่ต้องการจำแนก

# สร้างไดเรกทอรีสำหรับการบันทึกภาพ
plot = r"D:\cnn_lstm\picture"
os.makedirs(plot, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

class HandGestureDataset(Dataset):  # คลาสสำหรับจัดการชุดข้อมูลการเคลื่อนไหวมือ
    def __init__(self,
                 video_paths: List[str],
                 labels: List[int],
                 sequence_length: int = 15,
                 num_sequences: int = 1,
                 transform: Optional[object] = None):
        self.video_paths = video_paths  # เก็บเส้นทางของวิดีโอ
        self.labels = labels  # เก็บ labels ของข้อมูล
        self.sequence_length = sequence_length  # ความยาวของ sequence
        self.num_sequences = num_sequences  # จำนวน sequences ที่ต้องการ
        self.transform = transform  # การแปลงข้อมูลถ้ามี

    def __len__(self) -> int:  # คืนค่าจำนวนวิดีโอ
        return len(self.video_paths)

    def __getitem__(self, idx: int):  # ฟังก์ชันสำหรับดึงข้อมูลตามดัชนี
        try:
            frames = load_video(self.video_paths[idx])  # โหลดวิดีโอ
            if frames.size == 0:  # ถ้าวิดีโอว่าง
                # Return a zero tensor with the correct shape
                return (
                    torch.zeros((self.num_sequences, self.sequence_length, 3, 256, 192)),  # คืนค่า tensor ศูนย์
                    torch.tensor(self.labels[idx], dtype=torch.long)  # คืนค่า label ตามปกติ
                )

            # ตรวจสอบว่ามีเฟรมเพียงพอหรือไม่
            if len(frames) < self.sequence_length:  # ถ้ามีเฟรมน้อยกว่าที่ต้องการ
                # เพิ่มเฟรมซ้ำตามความจำเป็น
                last_frame = frames[-1]  # เฟรมสุดท้าย
                num_padding = self.sequence_length - len(frames)  # จำนวนเฟรมที่ต้องการเพิ่ม
                padding_frames = np.tile(last_frame, (num_padding, 1, 1, 1))  # ทำซ้ำเฟรมสุดท้าย
                frames = np.concatenate([frames, padding_frames])  # รวมเฟรมที่มีอยู่และเฟรมที่เพิ่ม

            elif len(frames) > self.sequence_length:  # ถ้ามีเฟรมมากกว่าที่ต้องการ
                # ใช้เฟรมที่ถูกเลือกแบบมีระยะห่าง
                indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)  # สร้างดัชนีที่มีระยะห่าง
                frames = frames[indices]  # เลือกเฟรมตามดัชนี

            # สร้าง sequences
            sequences = []  # รายการสำหรับเก็บ sequences
            for _ in range(self.num_sequences):  # วนรอบตามจำนวน sequences ที่ต้องการ
                sequence_frames = []  # รายการสำหรับเก็บเฟรมใน sequence
                for frame in frames:  # วนรอบผ่านแต่ละเฟรม
                    if self.transform:  # ถ้ามีการแปลง
                        frame = self.transform(frame)  # แปลงเฟรม
                    else:
                        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0  # แปลงเป็น tensor
                    sequence_frames.append(frame)  # เก็บเฟรมที่แปลงแล้ว
                sequences.append(torch.stack(sequence_frames))  # สร้าง tensor จากเฟรมที่แปลงแล้ว

            # Stack all sequences
            sequences_tensor = torch.stack(sequences)  # สร้าง tensor จาก sequences
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)  # แปลง label เป็น tensor

            return sequences_tensor, label_tensor  # คืนค่า sequences และ label
            
        except Exception as e:
            print(f"Error processing video {self.video_paths[idx]}: {e}")
            return (
                torch.zeros((self.num_sequences, self.sequence_length, 3, 256, 192)),
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

            # Conv Layer 2
            # Input size:  64.25 * 48.25 * 16
            # Spatial extend of each one (kernelConv size), F = 5
            # Slide size (strideConv), S = 2
            # Padding, P = 2
            ## Width: ((64.25 - 5 + 2 * 2) / 2) + 1 =  32.625 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((48.25 - 5 + 2 * 2) / 2) + 1 =   24.625 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 32
            ## Output Conv Layer1: 32.625 * 24.625 * 32
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm

            # Max Pooling Layer 2
            # Input size: 32.625 * 24.625 * 32
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 2
            ## Width: ((32.625 - 2) / 2) + 1 = 16.312
            ## High: ((24.625 - 2) / 2) + 1 = 12.312
            ## Depth: 32
            ### Output Max Pooling Layer 2: 16.312 * 12.312 * 32
            nn.MaxPool2d(2, 2),

            # Conv Layer 3
            # Input size: 16.312 * 12.312 * 32
            # Spatial extend of each one (kernelConv size), F = 5
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((16.312 - 5 + 2 * 2) / 2) + 1 =  8.656 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((12.312 - 5 + 2 * 2) / 2) + 1 =  6.656  #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 64
            ## Output Conv Layer1: 8.656 * 6.656 * 64
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=dropout),  # เพิ่ม Dropout2d หลัง BatchNorm

            # Max Pooling Layer 3
            # Input size: 8.656 * 6.656 * 64
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 2
            ## Width: ((8.656 - 2) / 2) + 1 = 4.328
            ## High: ((6.656 - 2) / 2) + 1 = 3.328
            ## Depth: 64
            ### Output Max Pooling Layer 2: 4.328 * 3.328 * 64
            nn.MaxPool2d(2, 2),
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

# input_size = 64 * 48 * 16 # for 1 layers CNN
# input_size = 16 * 12 * 32 # for 2 layers CNN
input_size = 4 * 3 * 64 # for 3 layers CNN
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


def objective(trial):
    # Print current trial number
    print(f"\n{'=' * 50}")
    print(f"Trial #{trial.number}")
    print(f"{'=' * 50}")

    # Hyperparameter optimization (Optuna)
    batch_size = trial.suggest_int('batch_size', 8, 256)
    hidden_units = trial.suggest_int('hidden_units', 100, 200)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 5)

    # manual
    num_epochs = 500
    dropout = 0.8
    learning_rate = 0.001

    # Print trial parameters
    print("\nTrial Parameters:")
    print(f"-- Batch Size: {batch_size}")
    print(f"-- Hidden Units: {hidden_units}")
    print(f"-- Hidden Layers: {hidden_layers}")
    print(f"-- Number of Epochs: {num_epochs}")
    print(f"-- Learning Rate: {learning_rate:.3f}")
    print(f"-- Dropout: {dropout:.1f}")

    print(f"{'=' * 50}")


    video_paths, labels = load_videos_from_folders(folder_paths)
    train_paths, test_paths, train_labels, test_labels = split_dataset(
        video_paths,
        labels,
        test_ratio=0.2, # แบ่ง train 80 test 20
        balance_method='oversample' # normalize ข้อมูลให้เท่ากันทุก class use 'oversample' or 'undersample'
    )

#### ส่วนนี้เดี๋ยวเอาไว้ทำ Augmentation
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    # ])

    num_sequences = 1 # เข้า LSTM ทีละ 1 set (15 frames)
    train_dataset = HandGestureDataset(train_paths, train_labels, num_sequences=num_sequences)
    test_dataset = HandGestureDataset(test_paths, test_labels, num_sequences=num_sequences)

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

    # Initialize loss tracking
    epoch_train_losses = defaultdict(list)
    epoch_val_losses = defaultdict(list)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        train_losses = []


        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            # Move data to device
            sequences, labels = sequences.to(device), labels.to(device)


            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Store loss
            train_losses.append(loss.item())

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Clear memory after each batch
            torch.cuda.empty_cache()

        train_acc = 100 * train_correct / train_total
        epoch_train_losses[epoch] = train_losses


        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_losses = []


        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())


                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Clear memory after each batch
                torch.cuda.empty_cache()

        val_acc = 100 * val_correct / val_total
        epoch_val_losses[epoch] = val_losses
        best_val_acc = max(best_val_acc, val_acc)

        # print(f"Epoch {epoch+1}")
        print(f"\nTraining Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print("-" * 30)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f'GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')


    # Print trial summary
    print(f"\nTrial #{trial.number} Summary:")
    print(f"{'=' * 30}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Parameters:")
    print(f"-- Batch Size: {batch_size}")
    print(f"-- Hidden Units: {hidden_units}")
    print(f"-- Hidden Layers: {hidden_layers}")
    print(f"-- Number of Epochs: {num_epochs}")
    print(f"-- Learning Rate: {learning_rate:.6f}")
    print(f"-- Dropout: {dropout:.6f}")
    print(f"{'=' * 50}")

    # Convert losses to numpy arrays for plotting
    train_losses_array = np.array([epoch_train_losses[i] for i in range(num_epochs)])
    val_losses_array = np.array([epoch_val_losses[i] for i in range(num_epochs)])

    # Create and save the loss plot
    plt.figure(figsize=(10, 6))

    plt.style.use('default')  # or try 'classic', 'bmh', 'ggplot'

    epochs = range(num_epochs)

    # Plot mean training loss and range
    plt.plot(epochs, np.mean(train_losses_array, axis=1), 'r-',
             label='Mean Training Loss', linewidth=2)
    train_std = np.std(train_losses_array, axis=1)
    plt.fill_between(epochs,
                     np.mean(train_losses_array, axis=1) - train_std,
                     np.mean(train_losses_array, axis=1) + train_std,
                     alpha=0.2, color='red', label='Training Loss Range')

    # Plot mean validation loss and range
    plt.plot(epochs, np.mean(val_losses_array, axis=1), 'b-',
             label='Mean Validation Loss', linewidth=2)
    val_std = np.std(val_losses_array, axis=1)
    plt.fill_between(epochs,
                     np.mean(val_losses_array, axis=1) - val_std,
                     np.mean(val_losses_array, axis=1) + val_std,
                     alpha=0.2, color='blue', label='Validation Loss Range')

    # Customize plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - Trial {trial.number}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)

    # Save plot
    save_path = os.path.join(plot, f'trial_{trial.number}_loss.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Store results
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
