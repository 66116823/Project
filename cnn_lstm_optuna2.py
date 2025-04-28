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
    select_frames,
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
                 max_missing_frames: int = 5,
                 num_sequences: int = 1,
                 transform: Optional[object] = None):
        self.video_paths = video_paths  # เก็บเส้นทางของวิดีโอ
        self.labels = labels  # เก็บ labels ของข้อมูล
        self.sequence_length = sequence_length  # ความยาวของ sequence
        self.max_missing_frames = max_missing_frames  # จำนวนเฟรมที่หายไปสูงสุด
        self.num_sequences = num_sequences  # จำนวน sequences ที่ต้องการ
        self.transform = transform  # การแปลงข้อมูลถ้ามี

    def standardize_sequences(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """Ensure each video returns exactly num_sequences sequences"""
        if len(sequences) == 0:  # ถ้าไม่มี sequences
            # สร้าง dummy sequences
            dummy_sequence = np.zeros((self.sequence_length, 192, 256, 3))  # สร้าง sequence ศูนย์
            return [dummy_sequence for _ in range(self.num_sequences)]  # คืนค่ารายการ dummy

        sequences = list(sequences)  # แปลงเป็น list ถ้าไม่ใช่

        if len(sequences) < self.num_sequences:  # ถ้ามี sequences น้อยกว่าที่ต้องการ
            # ทำซ้ำบาง sequences แบบสุ่ม
            while len(sequences) < self.num_sequences:
                sequences.append(sequences[random.randint(0, len(sequences) - 1)].copy())  # ทำซ้ำ
        elif len(sequences) > self.num_sequences:  # ถ้ามี sequences มากกว่าที่ต้องการ
            sequences = random.sample(sequences, self.num_sequences)  # เลือกแบบสุ่ม

        return sequences  # คืนค่า sequences ที่ได้

    def __len__(self) -> int:
        return len(self.video_paths)  # คืนค่าจำนวนวิดีโอ

    def __getitem__(self, idx: int):
        try:
            frames = load_video(self.video_paths[idx])  # โหลดวิดีโอ
            if frames.size == 0:  # ถ้าวิดีโอว่าง
                raise ValueError(f"Empty video: {self.video_paths[idx]}")  # แจ้งข้อผิดพลาด

            sequences = select_frames(  # เลือกเฟรมจากวิดีโอ
                frames,
                sequence_length=self.sequence_length,
                max_missing_frames=self.max_missing_frames
            )

            if not sequences:  # ถ้าไม่มี sequences ที่ถูกต้อง
                dummy_sequence = np.zeros((self.sequence_length, 192, 256, 3))  # สร้าง sequence ศูนย์
                sequences = [dummy_sequence]  # ตั้งค่าเป็น dummy sequence

            sequences = self.standardize_sequences(sequences)  # ทำให้ sequences เท่ากับจำนวนที่ต้องการ

            tensor_sequences = []  # รายการสำหรับเก็บ tensors
            for sequence in sequences:  # วนรอบผ่านแต่ละ sequence
                transformed_frames = []  # รายการสำหรับเก็บเฟรมที่แปลงแล้ว
                for frame in sequence:  # วนรอบผ่านแต่ละเฟรม
                    if self.transform:  # ถ้ามีการแปลง
                        frame = self.transform(frame)  # แปลงเฟรม
                    else:
                        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0  # แปลงเป็น tensor
                    transformed_frames.append(frame)  # เก็บเฟรมที่แปลงแล้ว
                tensor_sequences.append(torch.stack(transformed_frames))  # สร้าง tensor จากเฟรมที่แปลงแล้ว

            sequences_tensor = torch.stack(tensor_sequences)  # สร้าง tensor จาก sequences
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)  # แปลง label เป็น tensor

            return sequences_tensor, label_tensor  # คืนค่า sequences และ label


        except Exception as e:
            print(f"Error processing video {self.video_paths[idx]}: {e}")  # แสดงข้อผิดพลาด
            # Return a dummy tensor instead of raising an error
            return (
                # กว้าง 1024 สูง 768
                #  PyTorch (CHW - Channels, Height, Width)
                torch.zeros((self.num_sequences, self.sequence_length, 3, 192, 256)),  # คืนค่า tensor ศูนย์
                torch.tensor(self.labels[idx], dtype=torch.long)  # คืนค่า label ตามปกติ
            )


class CNN(nn.Module):  # คลาสสำหรับโมเดล CNN
    def __init__(self, activation):
        super(CNN, self).__init__()  # เรียกใช้ constructor ของคลาสแม่
        act_fn = {
            'ReLU': nn.ReLU(),  # ฟังก์ชัน activation: ReLU
            'Sigmoid': nn.Sigmoid(),  # ฟังก์ชัน activation: Sigmoid
            'tanh': nn.Tanh()  # ฟังก์ชัน activation: tanh
        }[activation]  # เลือกฟังก์ชัน activation ตามที่กำหนด

        # สร้างโมดูลสำหรับฟีเจอร์ของ CNN
        self.features = nn.Sequential(
            # Input size: 256*192*3
            # Spatial extend of each one (kernelConv size), F = 5
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((256 - 5 + 2 * 2) / 1) + 1 = 256 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((192 - 5 + 2 * 2) / 1) + 1 =  192 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 16
            ## Output Conv Layer1: 256 * 192 * 16

            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # Convolutional Layer 1
            act_fn,  # ฟังก์ชัน activation
            nn.BatchNorm2d(16),  # Batch Normalization

            # Max Pooling Layer 1
            # Input size: 256 * 192 * 16
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 1
            ## Width: ((256 - 2) / 2) + 1 = 128 #*# (( W2 - F ) / S ) + 1
            ## High: ((192 - 2) / 2) + 1 = 96 #*# (( H2 - F ) / S ) + 1
            ## Depth: 16
            ### Output Max Pooling Layer 1: 128 * 96 * 16
            nn.MaxPool2d(2, 2),  # Max Pooling Layer 1

            # Conv Layer 2
            # Input size:  128 * 96 * 16
            # Spatial extend of each one (kernelConv size), F = 5
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((128 - 5 + 2 * 2) / 1) + 1 =  128 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((96 - 5 + 2 * 2) / 1) + 1 =   96 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 32
            ## Output Conv Layer1: 128 * 96 * 32
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # Convolutional Layer 2
            act_fn,  # ฟังก์ชัน activation
            nn.BatchNorm2d(32),  # Batch Normalization

            # Max Pooling Layer 2
            # Input size: 128 * 96 * 32
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 2
            ## Width: ((128 - 2) / 2) + 1 = 64
            ## High: ((96 - 2) / 2) + 1 = 48
            ## Depth: 32
            ### Output Max Pooling Layer 2: 64 * 48* 32
            nn.MaxPool2d(2, 2),  # Max Pooling Layer 2

            # Conv Layer 3
            # Input size: 64 * 48 * 32
            # Spatial extend of each one (kernelConv size), F = 5
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((64 - 5 + 2 * 2) / 1) + 1 =  64 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((48 - 5 + 2 * 2) / 1) + 1 =   48 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 64
            ## Output Conv Layer1: 64 * 48 * 64
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # Convolutional Layer 3
            act_fn,  # ฟังก์ชัน activation
            nn.BatchNorm2d(64),  # Batch Normalization

            # Max Pooling Layer 3
            # Input size: 64 * 48 * 64
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 2
            ## Width: ((64 - 2) / 2) + 1 = 32
            ## High: ((48 - 2) / 2) + 1 = 24
            ## Depth: 64
            ### Output Max Pooling Layer 2: 32 * 24 * 64
            nn.MaxPool2d(2, 2),  # Max Pooling Layer 3

            # # Conv Layer 4
            # # Input size: 32 * 24 * 64
            # # Spatial extend of each one (kernelConv size), F = 5
            # # Slide size (strideConv), S = 1
            # # Padding, P = 2
            # ## Width: ((32 - 5 + 2 * 2) / 1) + 1 =  32 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            # ## High: ((24  - 5 + 2 * 2) / 1) + 1 = 24 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            # ## Depth: 128
            # ## Output Conv Layer1: 32 * 24 * 128
            # nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            # act_fn,
            # nn.BatchNorm2d(128),
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
        
# input_size = 128 * 96 * 16 # for 1 layers CNN
# input_size = 64 * 48* 32 # for 2 layers CNN
input_size = 32 * 24 * 64  # กำหนดขนาดอินพุตสำหรับ CNN 3 ชั้น
# input_size = 16 * 12 * 128 # for 4 layers CNN
# input_size = 8* 6* 256 # for 5 layers CNN

class CNN_LSTM(nn.Module):  # คลาสสำหรับโมเดล CNN-LSTM
    def __init__(self, hidden_units, hidden_layers, activation='ReLU'):
        super(CNN_LSTM, self).__init__()  # เรียกใช้ constructor ของคลาสแม่
        self.cnn = CNN(activation)  # สร้างโมเดล CNN
        self.lstm = nn.LSTM(  # สร้างโมเดล LSTM
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=hidden_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_units, num_classes)  # Fully Connected Layer
 
    def forward(self, x): # ฟังก์ชันสำหรับการประมวลผลข้อมูลผ่านโมเดล
        # x shape: (batch, num_sequences, seconds, channels, height, width)
        batch_size = x.size(0)  # ขนาดของแบตช์
        num_sequences = x.size(1)  # จำนวน sequences
        sequence_length = x.size(2)  # ความยาวของ sequence

        # เปลี่ยนรูปร่างสำหรับ CNN
        x = x.view(batch_size * num_sequences * sequence_length, *x.size()[3:])  # ปรับรูปร่างเข้ากับ CNN
        x = self.cnn(x)  # ประมวลผลด้วย CNN

        # เปลี่ยนรูปร่างสำหรับ LSTM
        x = x.view(batch_size * num_sequences, sequence_length, -1)  # ปรับรูปร่างเข้ากับ LSTM
        lstm_out, _ = self.lstm(x)  # ประมวลผลด้วย LSTM

        x = self.fc(lstm_out[:, -1, :])  # คำนวณผลลัพธ์สุดท้าย

        # คืนค่า
        x = x.view(batch_size, num_sequences, -1)  # คืนรูปร่างให้ถูกต้อง

        # คำนวณค่าเฉลี่ยของการทำนาย
        x = x.mean(dim=1)  # คืนค่าเฉลี่ยของ predictions

        return x  # คืนค่าผลลัพธ์


def objective(trial): # ฟังก์ชันสำหรับจัดการการทดลองของ Optuna
    # Print current trial number
    print(f"\n{'=' * 50}")
    print(f"Trial #{trial.number}")  # แสดงหมายเลขการทดลอง
    print(f"{'=' * 50}")

    # Hyperparameter optimization
    batch_size = trial.suggest_int('batch_size', 8, 128)  # แนะนำขนาดแบตช์
    activation = trial.suggest_categorical('activation', ['ReLU', 'Sigmoid', 'tanh'])  # แนะนำฟังก์ชัน activation
    hidden_units = trial.suggest_int('hidden_units', 50, 150)  # แนะนำจำนวนยูนิตใน LSTM
    hidden_layers = trial.suggest_int('hidden_layers', 1, 10)  # แนะนำจำนวนชั้นใน LSTM
    num_epochs = trial.suggest_int('num_epochs', 50, 200)  # แนะนำจำนวนยุคในการฝึก
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.099, log=True)  # แนะนำอัตราการเรียนรู้
    
    # Print trial parameters
    print("\nTrial Parameters:")
    print(f"-- Batch Size: {batch_size}")
    print(f"-- Activation Function: {activation}")
    print(f"-- Hidden Units: {hidden_units}")
    print(f"-- Hidden Layers: {hidden_layers}")
    print(f"-- Number of Epochs: {num_epochs}")
    print(f"-- Learning Rate: {learning_rate:.6f}")

    print(f"{'=' * 50}")

    # [Code for data loading and model setup remains the same]
    video_paths, labels = load_videos_from_folders(folder_paths) # โหลดวิดีโอจากโฟลเดอร์
    # แบ่งชุดข้อมูล
    train_paths, test_paths, train_labels, test_labels = split_dataset( 
        video_paths, labels, test_ratio=0.2)

    # ทำการแปลงข้อมูล
    transform = transforms.Compose([
        transforms.ToPILImage(),  # แปลงเป็น PIL Image
        transforms.RandomHorizontalFlip(p=0.5),  # พลิกภาพในแนวนอนแบบสุ่ม
        transforms.ToTensor(),  # แปลงเป็น Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ปกติภาพ
    ])

    num_sequences = 1  # กำหนดจำนวน sequences
    train_dataset = HandGestureDataset(train_paths, train_labels, num_sequences=num_sequences, transform=transform)  # สร้าง HandGestureDataset สำหรับการฝึก
    # val_dataset = HandGestureDataset(val_paths, val_labels, num_sequences=num_sequences, transform=transform)
    test_dataset = HandGestureDataset(test_paths, test_labels, num_sequences=num_sequences, transform=transform)  # สร้าง HandGestureDataset สำหรับการทดสอบ
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # สร้าง DataLoader สำหรับการฝึก
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # สร้าง DataLoader สำหรับการทดสอบ

    # สร้างโมเดล CNN-LSTM
    model = CNN_LSTM(
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        activation=activation
    ).to(device) # ส่งโมเดลไปยังอุปกรณ์

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # กำหนดออปติไมเซอร์ Adam
    criterion = nn.CrossEntropyLoss()  # กำหนดฟังก์ชันสูญเสีย CrossEntropy

    # Early stopping parameters
    patience = 5  # จำนวนยุคที่รอในการหยุด
    early_stopping_counter = 0  # ตัวนับสำหรับการหยุด

    # วนลูปการฝึกอบรม
    best_val_acc = 0.0  # ตัวแปรสำหรับเก็บ accuracy ที่ดีที่สุด
    print("\nTraining Progress:")  # แสดงหัวข้อการฝึก
    print("-" * 50)  # แสดงเส้นแบ่ง

    for epoch in range(num_epochs): # วนลูปตามจำนวนยุค
        # Training phase
        model.train()  # ตั้งค่าโมเดลเป็นโหมดการฝึก
        train_correct = 0  # ตัวแปรสำหรับเก็บจำนวนการทำนายที่ถูกต้องในการฝึก
        train_total = 0  # ตัวแปรสำหรับเก็บจำนวนข้อมูลทั้งหมดในการฝึก
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"): # วนลูปผ่าน DataLoader
            sequences, labels = sequences.to(device), labels.to(device) # ส่งข้อมูลไปยังอุปกรณ์

            optimizer.zero_grad()  # ล้าง gradients
            outputs = model(sequences)  # ประมวลผลข้อมูลผ่านโมเดล
            loss = criterion(outputs, labels)  # คำนวณ loss
            loss.backward()  # คำนวณ gradients
            optimizer.step()  # ปรับปรุงน้ำหนัก

            _, predicted = torch.max(outputs.data, 1)  # ทำนายคลาสจากผลลัพธ์
            train_total += labels.size(0)  # เพิ่มจำนวนข้อมูลทั้งหมด
            train_correct += (predicted == labels).sum().item()  # คำนวณจำนวนการทำนายที่ถูกต้อง

        train_acc = 100 * train_correct / train_total  # คำนวณ accuracy ของการฝึก
        
        # เฟสการตรวจสอบ
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

        # ตรวจสอบการหยุดการฝึก
        if val_acc > best_val_acc:  # ถ้า accuracy ใหม่สูงกว่า
            best_val_acc = val_acc  # อัปเดตค่า accuracy ที่ดีที่สุด
            early_stopping_counter = 0  # รีเซ็ตตัวนับเมื่อมีการปรับปรุง
        else:
            early_stopping_counter += 1  # ไม่มีการปรับปรุง เพิ่มตัวนับ

        # ถ้าไม่มีการปรับปรุงใน 'patience' epochs หยุดการฝึก
        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy.")  # แสดงข้อความหยุด
            break  # หยุดการฝึก

    # # Final test evaluation
    # model.eval()
    # test_correct = 0
    # test_total = 0
    # with torch.no_grad():
    #     for sequences, labels in test_loader:
    #         sequences, labels = sequences.to(device), labels.to(device)
    #         outputs = model(sequences)
    #         _, predicted = torch.max(outputs.data, 1)
    #         test_total += labels.size(0)
    #         test_correct += (predicted == labels).sum().item()
    #
    # test_acc = 100 * test_correct / test_total

    # Print trial summary
    print(f"\nTrial #{trial.number} Summary:")  # แสดงหัวข้อสรุป
    print(f"{'=' * 30}")  # แสดงเส้นแบ่ง
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")  # แสดง accuracy ที่ดีที่สุด
    # print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Parameters:")  # แสดงหัวข้อพารามิเตอร์
    print(f"- Batch Size: {batch_size}")  # แสดงขนาดแบตช์
    print(f"- Number of Epochs: {num_epochs}")  # แสดงจำนวนยุค
    print(f"- Learning Rate: {learning_rate:.6f}")  # แสดงอัตราการเรียนรู้
    print(f"{'=' * 50}")  # แสดงเส้นแบ่ง

    # Store results
    # trial.set_user_attr('test_accuracy', test_acc)
    # เก็บผลลัพธ์
    trial.set_user_attr('best_val_accuracy', best_val_acc)  # เก็บ accuracy ที่ดีที่สุดใน trial

    return best_val_acc  # คืนค่า accuracy ที่ดีที่สุด

def main():  # ฟังก์ชันหลัก
    # ข้อมูลเกี่ยวกับ GPU
    print(f"Using device: {device}")  # แสดงอุปกรณ์ที่ใช้
    print(f"CUDA available: {torch.cuda.is_available()}")  # แจ้งว่ามี CUDA หรือไม่
    if torch.cuda.is_available():  # ถ้ามี CUDA
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")  # แสดงชื่อของ GPU
        
    # การตั้งค่าของ Optuna
    study = optuna.create_study(direction='maximize')  # สร้างการศึกษาใหม่เพื่อหาค่ามากที่สุด
    study.optimize(objective, n_trials=10, catch=(Exception,))  # เรียกใช้ฟังก์ชัน objective สำหรับการทดลอง

    print("Best trial:")  # แสดงรายละเอียดของการทดลองที่ดีที่สุด
    trial = study.best_trial  # ดึงการทดลองที่ดีที่สุด
    print("  Value: {}".format(trial.value))  # แสดงค่าที่ดีที่สุด
    print("  Params: ")  # แสดงพารามิเตอร์ของการทดลองที่ดีที่สุด
    for key, value in trial.params.items():  # วนรอบพารามิเตอร์
        print("    {}: {}".format(key, value))  # แสดงชื่อและค่าของพารามิเตอร์


if __name__ == "__main__":  # ถ้ารันไฟล์นี้เป็นโปรแกรมหลัก
    main()  # เรียกใช้ฟังก์ชันหลัก
