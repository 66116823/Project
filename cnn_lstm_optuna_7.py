import torch  # นำเข้า PyTorch สำหรับการสร้างโมเดลและการคำนวณ
import torch.nn as nn  # นำเข้าโมดูล neural network จาก PyTorch
from torch.utils.data import Dataset, DataLoader  # นำเข้า Dataset และ DataLoader สำหรับการจัดการข้อมูล
import torchvision.transforms as transforms  # นำเข้า transforms สำหรับการแปลงข้อมูลภาพ
import optuna  # นำเข้า Optuna สำหรับการทำ hyperparameter optimization
from tqdm import tqdm  # นำเข้า tqdm สำหรับการแสดง progress bar
import numpy as np  # นำเข้า NumPy สำหรับการคำนวณเชิงตัวเลข
import random  # นำเข้า random สำหรับการสุ่ม
from typing import List, Optional  # นำเข้า typing สำหรับการกำหนดประเภทข้อมูล

# นำเข้าสำหรับการประมวลผลวิดีโอ
from cnn_preprocessing import (
    load_video,
    # select_frames,
    split_dataset,
    folder_paths,
    load_videos_from_folders
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # กำหนดอุปกรณ์สำหรับการประมวลผล (GPU หรือ CPU)
num_classes = 4  # จำนวนคลาสที่ต้องการจำแนก

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
                # คืนค่า tensor ศูนย์ที่มีรูปร่างที่ถูกต้อง
                return (
                    torch.zeros((self.num_sequences, self.sequence_length, 3, 256, 192)),
                    torch.tensor(self.labels[idx], dtype=torch.long)
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
            print(f"Error processing video {self.video_paths[idx]}: {e}")  # แสดงข้อผิดพลาด
            return (
                torch.zeros((self.num_sequences, self.sequence_length, 3, 256, 192)),  # คืนค่า tensor ศูนย์
                torch.tensor(self.labels[idx], dtype=torch.long)  # คืนค่า label ตามปกติ
            )



class CNN(nn.Module):  # คลาสสำหรับโมเดล CNN
    def __init__(self, activation, dropout):  # กำหนดค่าดรอปเอาต์และฟังก์ชัน activation
        super(CNN, self).__init__()  # เรียกใช้ constructor ของคลาสแม่
        act_fn = {
            'ReLU': nn.ReLU(),  # ฟังก์ชัน activation: ReLU
            'Sigmoid': nn.Sigmoid(),  # ฟังก์ชัน activation: Sigmoid
            'tanh': nn.Tanh()  # ฟังก์ชัน activation: tanh
        }[activation]  # เลือกฟังก์ชัน activation ตามที่กำหนด

        # สร้างโมดูลสำหรับฟีเจอร์ของ CNN
        self.features = nn.Sequential(
            # Input size: 256*192*3
            # Spatial extend of each one (kernelConv size), F = 7
            # Slide size (strideConv), S = 2
            # Padding, P = 2
            ## Width: ((256 - 7 + 2 * 2) / 2) + 1 = 127.5 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((192 - 7 + 2 * 2) / 2) + 1 =  95.5 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 16
            ## Output Conv Layer1: 127.5  * 95.5 * 16

            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=2),  # Convolutional Layer 1
            act_fn,  # ฟังก์ชัน activation
            nn.BatchNorm2d(16),  # Batch Normalization
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
            nn.MaxPool2d(2, 2), # Max Pooling Layer 1

            # Conv Layer 2
            # Input size:  63.75 * 47.75 * 16
            # Spatial extend of each one (kernelConv size), F = 7
            # Slide size (strideConv), S = 2
            # Padding, P = 2
            ## Width: ((63.75 - 7 + 2 * 2) / 2) + 1 =  31.375 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((47.75  - 7 + 2 * 2) / 2) + 1 =  23.372  #*# W2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 32
            ## Output Conv Layer1: 31.375 * 23.372 * 32
            nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=2),  # Convolutional Layer 2
            act_fn,  # ฟังก์ชัน activation
            nn.BatchNorm2d(32),  # Batch Normalization
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
            nn.MaxPool2d(2, 2), # Max Pooling Layer 2

            # Conv Layer 3
            # Input size: 15.6875 * 11.686 * 32
            # Spatial extend of each one (kernelConv size), F = 7
            # Slide size (strideConv), S = 2
            # Padding, P = 2
            ## Width: (( 15.6875  - 7+ 2 * 2) / 2) + 1 = 7.34375  #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((11.686 - 7+ 2 * 2) / 2) + 1 = 5.343 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 64
            ## Output Conv Layer1: 7.34375  * 5.343 * 64
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=2),  # Convolutional Layer 3
            act_fn,  # ฟังก์ชัน activation
            nn.BatchNorm2d(64),  # Batch Normalization
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
            nn.MaxPool2d(2, 2), # Max Pooling Layer 3

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

    def forward(self, x):  # ฟังก์ชันสำหรับการประมวลผลข้อมูลผ่านโมเดล
        return self.features(x)  # คืนค่าผลลัพธ์จากฟีเจอร์

# input_size = 63 * 47 * 16 # for 1 layers CNN
# input_size = 15 * 11 * 32 # for 2 layers CNN
input_size = 3 * 2 * 64 # for 3 layers CNN 


class CNN_LSTM(nn.Module):  # คลาสสำหรับโมเดล CNN-LSTM
    def __init__(self, hidden_units, hidden_layers, activation, dropout):  # กำหนดจำนวนยูนิตและชั้นใน LSTM
        super(CNN_LSTM, self).__init__()  # เรียกใช้ constructor ของคลาสแม่
        self.cnn = CNN(  # สร้างโมเดล CNN
            activation=activation,
            dropout=dropout
        )

        # สร้างโมเดล LSTM
        self.lstm = nn.LSTM(
            input_size=input_size ,
            hidden_size=hidden_units,
            num_layers=hidden_layers,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=dropout) # สร้าง Dropout Layer
        self.fc = nn.Linear(hidden_units, num_classes) ## numclasses คือจำนวน label ที่จะให้ทำนายตอนท้ายสุด

    def forward(self, x): # ฟังก์ชันสำหรับการประมวลผลข้อมูลผ่านโมเดล
        # x shape: (batch, num_sequences, seconds, channels, height, width)
        batch_size = x.size(0)  # ขนาดของแบตช์
        num_sequences = x.size(1)  # จำนวน sequences
        sequence_length = x.size(2)  # ความยาวของ sequence
        
        # Reshape for CNN
        x = x.view(batch_size * num_sequences * sequence_length, *x.size()[3:])  # ปรับรูปร่างเข้ากับ CNN
        x = self.cnn(x)  # ประมวลผลด้วย CNN

        # Reshape for LSTM
        x = x.view(batch_size * num_sequences, sequence_length, -1)  # ปรับรูปร่างเข้ากับ LSTM
        lstm_out, _ = self.lstm(x)  # ประมวลผลด้วย LSTM

        # Take final output
        # x = self.fc(lstm_out[:, -1, :])
        x = self.fc(self.dropout(lstm_out[:, -1, :]))  # เพิ่ม dropout ก่อน fully connected layer

        # Reshape back
        x = x.view(batch_size, num_sequences, -1)  # คืนรูปร่างให้ถูกต้อง

        # Average predictions
        x = x.mean(dim=1)  # คืนค่าเฉลี่ยของ predictions

        return x # คืนค่าผลลัพธ์


def objective(trial):  # ฟังก์ชันสำหรับจัดการการทดลองของ Optuna
    print(f"\n{'=' * 50}")
    print(f"Trial #{trial.number}")  # แสดงหมายเลขการทดลอง
    print(f"{'=' * 50}")

    # Hyperparameter optimization
    batch_size = trial.suggest_int('batch_size', 8, 256)  # แนะนำขนาดแบตช์
    activation = trial.suggest_categorical('activation', ['ReLU', 'Sigmoid', 'tanh'])  # แนะนำฟังก์ชัน activation
    hidden_units = trial.suggest_int('hidden_units', 50, 150)  # แนะนำจำนวนยูนิตใน LSTM
    hidden_layers = trial.suggest_int('hidden_layers', 1, 10)  # แนะนำจำนวนชั้นใน LSTM
    num_epochs = trial.suggest_int('num_epochs', 200, 1000)  # แนะนำจำนวนยุคในการฝึก
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.099, log=True)  # แนะนำอัตราการเรียนรู้
    dropout = trial.suggest_float('dropout', 0.1, 0.7, log=True)  # แนะนำค่าดรอปเอาต์



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
    video_paths, labels = load_videos_from_folders(folder_paths)  # โหลดวิดีโอจากโฟลเดอร์
    train_paths, test_paths, train_labels, test_labels = split_dataset(  # แบ่งชุดข้อมูล
        video_paths, labels, test_ratio=0.2)  # แบ่งเป็น 80% สำหรับการฝึก และ 20% สำหรับการทดสอบ

    transform = transforms.Compose([  # สร้างการแปลงข้อมูล
        transforms.ToPILImage(),  # แปลงเป็น PIL Image
        transforms.RandomHorizontalFlip(p=0.5),  # พลิกภาพในแนวนอนแบบสุ่ม
        transforms.ToTensor(),  # แปลงเป็น Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ปกติภาพ
    ])
    
    num_sequences = 1  # กำหนดจำนวน sequences
    train_dataset = HandGestureDataset(train_paths, train_labels, num_sequences=num_sequences, transform=transform)  # สร้าง HandGestureDataset สำหรับการฝึก
    test_dataset = HandGestureDataset(test_paths, test_labels, num_sequences=num_sequences, transform=transform)  # สร้าง HandGestureDataset สำหรับการทดสอบ

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # สร้าง DataLoader สำหรับการฝึก
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # สร้าง DataLoader สำหรับการทดสอบ

    model = CNN_LSTM(  # สร้างโมเดล CNN-LSTM
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout
    ).to(device)  # ส่งโมเดลไปยังอุปกรณ์

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # กำหนดออปติไมเซอร์ Adam
    criterion = nn.CrossEntropyLoss()  # กำหนดฟังก์ชันสูญเสีย CrossEntropy

    # # Early stopping parameters
    # patience = 5
    # early_stopping_counter = 0

    # Training loop
    # วนลูปการฝึกอบรม
    best_val_acc = 0.0  # ตัวแปรสำหรับเก็บ accuracy ที่ดีที่สุด
    print("\nTraining Progress:")  # แสดงหัวข้อการฝึก
    print("-" * 50)  # แสดงเส้นแบ่ง

    for epoch in range(num_epochs):  # วนลูปตามจำนวนยุค
        # Training phase
        model.train()  # ตั้งค่าโมเดลเป็นโหมดการฝึก
        train_correct = 0  # ตัวแปรสำหรับเก็บจำนวนการทำนายที่ถูกต้องในการฝึก
        train_total = 0  # ตัวแปรสำหรับเก็บจำนวนข้อมูลทั้งหมดในการฝึก
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):  # วนลูปผ่าน DataLoader
            sequences, labels = sequences.to(device), labels.to(device)  # ส่งข้อมูลไปยังอุปกรณ์

            optimizer.zero_grad()  # ล้าง gradients
            outputs = model(sequences)  # ประมวลผลข้อมูลผ่านโมเดล
            loss = criterion(outputs, labels)  # คำนวณ loss
            loss.backward()  # คำนวณ gradients
            optimizer.step()  # ปรับปรุงน้ำหนัก

            # คำนวณ accuracy
            _, predicted = torch.max(outputs.data, 1)  # ทำนายคลาสจากผลลัพธ์
            train_total += labels.size(0)  # เพิ่มจำนวนข้อมูลทั้งหมด
            train_correct += (predicted == labels).sum().item()  # คำนวณจำนวนการทำนายที่ถูกต้อง

        train_acc = 100 * train_correct / train_total  # คำนวณ accuracy ของการฝึก

        # Validation phase
        model.eval()  # ตั้งค่าโมเดลเป็นโหมดตรวจสอบ
        val_correct = 0  # ตัวแปรสำหรับเก็บจำนวนการทำนายที่ถูกต้องในการตรวจสอบ
        val_total = 0  # ตัวแปรสำหรับเก็บจำนวนข้อมูลทั้งหมดในการตรวจสอบ
        with torch.no_grad():  # ปิดการคำนวณ gradients
            for sequences, labels in test_loader:  # วนลูปผ่าน DataLoader
                sequences, labels = sequences.to(device), labels.to(device)  # ส่งข้อมูลไปยังอุปกรณ์
                outputs = model(sequences)  # ประมวลผลข้อมูลผ่านโมเดล
                _, predicted = torch.max(outputs.data, 1)  # ทำนายคลาสจากผลลัพธ์
                val_total += labels.size(0)  # เพิ่มจำนวนข้อมูลทั้งหมด
                val_correct += (predicted == labels).sum().item()  # คำนวณจำนวนการทำนายที่ถูกต้อง

        val_acc = 100 * val_correct / val_total  # คำนวณ accuracy ของการตรวจสอบ
        best_val_acc = max(best_val_acc, val_acc)  # ติดตาม accuracy ที่ดีที่สุด

        # print(f"Epoch {epoch+1}")
        print(f"\nTraining Accuracy: {train_acc:.2f}%")  # แสดง accuracy ของการฝึก
        print(f"Validation Accuracy: {val_acc:.2f}%")  # แสดง accuracy ของการตรวจสอบ
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")  # แสดง accuracy ที่ดีที่สุด
        print("-" * 30)  # แสดงเส้นแบ่ง

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
    print(f"\nTrial #{trial.number} Summary:")  # แสดงหัวข้อสรุป
    print(f"{'=' * 30}")  # แสดงเส้นแบ่ง
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")  # แสดง accuracy ที่ดีที่สุด
    # print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Parameters:")  # แสดงหัวข้อพารามิเตอร์
    print(f"-- Batch Size: {batch_size}")  # แสดงขนาดแบตช์
    print(f"-- Activation Function: {activation}")  # แสดงฟังก์ชัน activation
    print(f"-- Hidden Units: {hidden_units}")  # แสดงจำนวนยูนิตใน LSTM
    print(f"-- Hidden Layers: {hidden_layers}")  # แสดงจำนวนชั้นใน LSTM
    print(f"-- Number of Epochs: {num_epochs}")  # แสดงจำนวนยุคในการฝึก
    print(f"-- Learning Rate: {learning_rate:.6f}")  # แสดงอัตราการเรียนรู้
    print(f"-- Dropout: {dropout:.6f}")  # แสดงค่าดรอปเอาต์
    print(f"{'=' * 50}")  # แสดงเส้นแบ่ง

    # Store results
    # trial.set_user_attr('test_accuracy', test_acc)
    trial.set_user_attr('best_val_accuracy', best_val_acc) # เก็บ accuracy ที่ดีที่สุดใน trial

    return best_val_acc # คืนค่า accuracy ที่ดีที่สุด


def main():  # ฟังก์ชันหลัก
    # ข้อมูลเกี่ยวกับ GPU
    print(f"Using device: {device}")  # แสดงอุปกรณ์ที่ใช้
    print(f"CUDA available: {torch.cuda.is_available()}")  # แจ้งว่ามี CUDA หรือไม่
    if torch.cuda.is_available():  # ถ้ามี CUDA
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")  # แสดงชื่อของ GPU

    # การตั้งค่าของ Optuna
    study = optuna.create_study(direction='maximize')  # สร้างการศึกษาใหม่เพื่อหาค่ามากที่สุด
    study.optimize(objective, n_trials=10, catch=(Exception,))  # เรียกใช้ฟังก์ชัน objective สำหรับการทดลอง

    print("\nBest Trial Results:")  # แสดงรายละเอียดของการทดลองที่ดีที่สุด
    print("=" * 50)  # แสดงเส้นแบ่ง
    trial = study.best_trial  # ดึงการทดลองที่ดีที่สุด
    # หาหมายเลขการทดลองที่ดีที่สุด
    best_trial_number = None  # ตัวแปรสำหรับเก็บหมายเลขการทดลองที่ดีที่สุด
    for trial_idx, t in enumerate(study.trials):  # วนรอบทุกการทดลอง
        if t.number == trial.number:  # ถ้าหมายเลขการทดลองตรงกัน
            best_trial_number = trial_idx + 1  # เก็บหมายเลขการทดลองที่ดีที่สุด
            break

    print(f"Best Trial Number: {best_trial_number} (Trial #{trial.number})")  # แสดงหมายเลขการทดลองที่ดีที่สุด
    print(f"Best Accuracy: {trial.value:.2f}%")  # แสดงค่าที่ดีที่สุด
    print("\nBest Parameters:")  # แสดงพารามิเตอร์ของการทดลองที่ดีที่สุด
    for key, value in trial.params.items():  # วนรอบพารามิเตอร์
        print("    {}: {}".format(key, value))  # แสดงชื่อและค่าของพารามิเตอร์


if __name__ == "__main__":  # ถ้ารันไฟล์นี้เป็นโปรแกรมหลัก
    main()  # เรียกใช้ฟังก์ชันหลัก
