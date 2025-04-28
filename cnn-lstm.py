import torch  # นำเข้า PyTorch สำหรับการสร้างโมเดลและการคำนวณ
import torch.nn as nn  # นำเข้าโมดูล neural network จาก PyTorch
from torch.utils.data import Dataset, DataLoader  # นำเข้า Dataset และ DataLoader สำหรับการจัดการข้อมูล
import numpy as np  # นำเข้า NumPy สำหรับการคำนวณเชิงตัวเลข
from typing import List, Optional  # นำเข้า typing สำหรับการกำหนดประเภทข้อมูล
import gc  # นำเข้า gc สำหรับการจัดการหน่วยความจำ
import os  # นำเข้า os สำหรับการจัดการไฟล์และโฟลเดอร์
import matplotlib.pyplot as plt  # นำเข้า Matplotlib สำหรับการสร้างกราฟ
from typing import List, Dict  # นำเข้า typing สำหรับการกำหนดประเภทข้อมูล
import time
import time  # นำเข้า time สำหรับการวัดเวลา
import psutil  # นำเข้า psutil สำหรับการตรวจสอบการใช้งานหน่วยความจำ
import numpy as np
from tqdm import tqdm  # นำเข้า tqdm สำหรับการแสดง progress bar
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score, classification_report, confusion_matrix # นำเข้า metrics จาก scikit-learn สำหรับการประเมินผล
import seaborn as sns  # นำเข้า Seaborn สำหรับการสร้างกราฟที่สวยงาม

from cnn_preprocessing import VideoPreprocessor  # นำเข้า VideoPreprocessor สำหรับการประมวลผลวิดีโอ
preprocessor = VideoPreprocessor()  # สร้างอ็อบเจ็กต์ของ VideoPreprocessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # กำหนดอุปกรณ์สำหรับการประมวลผล (GPU หรือ CPU)
num_classes = 4  # จำนวนคลาสที่ต้องการจำแนก

# สร้างไดเรกทอรีสำหรับการบันทึกกราฟ
plot = r"D:\cnn_lstm\Plot"
os.makedirs(plot, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

class TrainingVisualizer:  # คลาสสำหรับการแสดงผลการฝึกอบรม
    def __init__(self, save_dir: str, dropout: float):
        """
        กำหนดค่าเริ่มต้นสำหรับ TrainingVisualizer

        Args:
            save_dir: โฟลเดอร์สำหรับบันทึกกราฟ
            dropout: ค่าดรอปเอาต์ที่ใช้ในการฝึกอบรม
        """
        self.save_dir = save_dir  # กำหนดที่เก็บไฟล์
        self.dropout = dropout  # กำหนดค่าดรอปเอาต์
        os.makedirs(save_dir, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

        # เก็บเมตริกในการฝึกอบรม
        self.metrics = {
            'train_loss': [],  # รายการสำหรับเก็บค่า loss ของการฝึกอบรม
            'val_loss': [],  # รายการสำหรับเก็บค่า loss ของการตรวจสอบ
            'train_acc': [],  # รายการสำหรับเก็บค่า accuracy ของการฝึกอบรม
            'val_acc': [],  # รายการสำหรับเก็บค่า accuracy ของการตรวจสอบ
            'gpu_memory': [],  # รายการสำหรับเก็บการใช้งานหน่วยความจำ GPU
            'batch_times': [],  # รายการสำหรับเก็บเวลาที่ใช้ในการประมวลผลแต่ละแบตช์
            'epoch_times': []  # รายการสำหรับเก็บเวลาที่ใช้ในการฝึกอบรมแต่ละยุค
        }

        # ใช้สไตล์เริ่มต้นแทน seaborn
        plt.style.use('default')

    def update_metrics(self, epoch_metrics: Dict):
        """อัปเดตเมตริกที่เก็บไว้ด้วยค่าที่ใหม่"""
        for key, value in epoch_metrics.items():  # วนรอบค่าต่างๆ
            if key in self.metrics:  # ถ้ากุญแจอยู่ในเมตริก
                self.metrics[key].append(value)  # เพิ่มค่าในรายการ

    def plot_all_metrics(self):
        """สร้างกราฟสำหรับเมตริกทั้งหมด"""
        # ตรวจสอบว่ามีข้อมูลเพื่อสร้างกราฟหรือไม่
        if not self.metrics['train_loss']:  # ถ้าไม่มีเมตริก
            print("No metrics to plot.")  # แจ้งว่าไม่มีเมตริกให้แสดง
            return

        # สร้างกราฟ loss ของการฝึกอบรมและการตรวจสอบ
        plt.figure(figsize=(10, 6))  # กำหนดขนาดของกราฟ
        epochs = range(1, len(self.metrics['train_loss']) + 1)  # สร้างช่วงยุค
        plt.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)  # กราฟ loss ของการฝึกอบรม
        plt.plot(epochs, self.metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)  # กราฟ loss ของการตรวจสอบ
        plt.title('Training and Validation Loss')  # ตั้งชื่อกราฟ
        plt.xlabel('Epochs')  # ตั้งชื่อแกน x
        plt.ylabel('Loss')  # ตั้งชื่อแกน y
        plt.legend()  # แสดงตำนาน
        plt.grid(True, alpha=0.3)  # แสดงกริด
        plt.savefig(os.path.join(self.save_dir, f'loss_dropout_{self.dropout}.png'))  # บันทึกกราฟ
        plt.close()  # ปิดกราฟ

        # สร้างกราฟ accuracy ของการฝึกอบรมและการตรวจสอบ
        plt.figure(figsize=(10, 6))  # กำหนดขนาดของกราฟ
        plt.plot(epochs, self.metrics['train_acc'], 'b-', label='Training Accuracy', linewidth=2)  # กราฟ accuracy ของการฝึกอบรม
        plt.plot(epochs, self.metrics['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)  # กราฟ accuracy ของการตรวจสอบ
        plt.title('Training and Validation Accuracy')  # ตั้งชื่อกราฟ
        plt.xlabel('Epochs')  # ตั้งชื่อแกน x
        plt.ylabel('Accuracy (%)')  # ตั้งชื่อแกน y
        plt.legend()  # แสดงตำนาน
        plt.grid(True, alpha=0.3)  # แสดงกริด
        plt.savefig(os.path.join(self.save_dir, f'accuracy_dropout_{self.dropout}.png'))  # บันทึกกราฟ
        plt.close()  # ปิดกราฟ

        # สร้างกราฟการใช้งานหน่วยความจำ GPU
        plt.figure(figsize=(10, 6))  # กำหนดขนาดของกราฟ
        plt.plot(epochs, self.metrics['gpu_memory'], 'g-', linewidth=2)  # กราฟการใช้งานหน่วยความจำ GPU
        plt.title('GPU Memory Usage Over Time')  # ตั้งชื่อกราฟ
        plt.xlabel('Epochs')  # ตั้งชื่อแกน x
        plt.ylabel('Memory Usage (GB)')  # ตั้งชื่อแกน y
        plt.grid(True, alpha=0.3)  # แสดงกริด
        plt.savefig(os.path.join(self.save_dir, f'gpu_memory_dropout_{self.dropout}.png'))  # บันทึกกราฟ
        plt.close()  # ปิดกราฟ

        # สร้างกราฟเวลาในการฝึกอบรมแต่ละยุค
        plt.figure(figsize=(10, 6))  # กำหนดขนาดของกราฟ
        plt.plot(epochs, self.metrics['epoch_times'], 'b-', linewidth=2)  # กราฟเวลาในการฝึกอบรม
        plt.title('Epoch Training Times')  # ตั้งชื่อกราฟ
        plt.xlabel('Epochs')  # ตั้งชื่อแกน x
        plt.ylabel('Time (seconds)')  # ตั้งชื่อแกน y
        plt.grid(True, alpha=0.3)  # แสดงกริด
        plt.savefig(os.path.join(self.save_dir, f'epoch_times_dropout_{self.dropout}.png'))  # บันทึกกราฟ
        plt.close()  # ปิดกราฟ

        # สร้างกราฟเวลาเฉลี่ยในการประมวลผลแต่ละแบตช์
        plt.figure(figsize=(10, 6))  # กำหนดขนาดของกราฟ
        plt.plot(epochs, self.metrics['batch_times'], 'r-', linewidth=2)  # กราฟเวลาเฉลี่ยในการประมวลผล
        plt.title('Average Batch Processing Times')  # ตั้งชื่อกราฟ
        plt.xlabel('Epochs')  # ตั้งชื่อแกน x
        plt.ylabel('Time (seconds)')  # ตั้งชื่อแกน y
        plt.grid(True, alpha=0.3)  # แสดงกริด
        plt.savefig(os.path.join(self.save_dir, f'batch_times_dropout_{self.dropout}.png'))  # บันทึกกราฟ
        plt.close()  # ปิดกราฟ


class HandGestureDataset(Dataset):  # คลาสสำหรับจัดการชุดข้อมูลการเคลื่อนไหวมือ
    def __init__(self,
                 sequences: np.ndarray,
                 labels: np.ndarray,
                 num_sequences: int = 1,
                 transform: Optional[object] = None):
        self.sequences = sequences  # เก็บ sequences ของข้อมูล
        self.labels = labels  # เก็บ labels ของข้อมูล
        self.num_sequences = num_sequences  # จำนวน sequences ที่ต้องการ
        self.transform = transform  # การแปลงข้อมูลถ้ามี

    def __len__(self) -> int:
        return len(self.sequences)  # คืนค่าจำนวน sequences

    def __getitem__(self, idx: int):
        try:
            # ดึง sequence และแปลงเป็น tensor
            sequence = self.sequences[idx]

            # ใช้ transform ถ้ามี
            if self.transform:
                transformed_frames = []  # รายการสำหรับเก็บกรอบที่แปลงแล้ว
                for frame in sequence:  # วนรอบกรอบใน sequence
                    transformed_frames.append(self.transform(frame))  # แปลงกรอบ
                sequence = np.stack(transformed_frames)  # สร้าง numpy array จากกรอบที่แปลงแล้ว

            # แปลงเป็น tensor และ normalize
            sequence_tensor = torch.from_numpy(sequence.transpose(0, 3, 1, 2)).float() / 255.0  # นำเข้าเป็น tensor

            # เปลี่ยนรูปร่างสำหรับหลาย sequences ถ้าจำเป็น
            sequences_tensor = sequence_tensor.unsqueeze(0).repeat(self.num_sequences, 1, 1, 1, 1)  # ทำซ้ำตามจำนวน sequences

            # ดึง label
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)  # แปลง label เป็น tensor

            return sequences_tensor, label_tensor  # คืนค่า sequences และ label

        except Exception as e:
            print(f"Error processing sequence {idx}: {e}")  # แสดงข้อความเมื่อเกิดข้อผิดพลาด
            return (
                torch.zeros((self.num_sequences, 15, 3, 256, 192)),  # คืนค่า tensor ศูนย์ถ้าเกิดข้อผิดพลาด
                torch.tensor(self.labels[idx], dtype=torch.long)  # คืนค่า label ตามปกติ
            )
            

class CNN(nn.Module):  # คลาสสำหรับโมเดล CNN
    def __init__(self, dropout):
        super(CNN, self).__init__()  # เรียกใช้ constructor ของคลาสแม่

        self.features = nn.Sequential(  # สร้างโมดูลสำหรับฟีเจอร์ของ CNN
            # Input size: 256*192*3
            # Spatial extend of each one (kernelConv size), F = 5
            # Slide size (strideConv), S = 2
            # Padding, P = 2
            ## Width: ((256 - 5 + 2 * 2) / 2) + 1 = 128.5 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((192 - 5 + 2 * 2) / 2) + 1 =  96.5 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 16
            ## Output Conv Layer1:  128.5 * 96.5 * 16
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # Convolutional Layer 1
            nn.ReLU(),  # Activation Function
            nn.BatchNorm2d(16),  # Batch Normalization
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
            nn.MaxPool2d(2, 2),  # Max Pooling Layer 1
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

class CNN_LSTM(nn.Module):  # คลาสสำหรับโมเดล CNN-LSTM
    def __init__(self, hidden_units, hidden_layers, dropout):
        super(CNN_LSTM, self).__init__()  # เรียกใช้ constructor ของคลาสแม่
        self.cnn = CNN(
            dropout=dropout
        ) # สร้างโมเดล CNN

        self.lstm = nn.LSTM(  # สร้างโมเดล LSTM
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=hidden_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)  # สร้าง Dropout Layer
        self.fc = nn.Linear(hidden_units, num_classes) ## numclasses คือจำนวน label ที่จะให้ทำนายตอนท้ายสุด

    def forward(self, x):  # ฟังก์ชันสำหรับการประมวลผลข้อมูลผ่านโมเดล
        # x shape: (batch, num_sequences, seconds, channels, height, width)
        batch_size = x.size(0)  # ขนาดของแบตช์
        num_sequences = x.size(1)  # จำนวน sequences
        sequence_length = x.size(2)  # ความยาวของ sequence

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


class BenchmarkMetrics:  # คลาสสำหรับเก็บค่าการวัดประสิทธิภาพ
    def __init__(self):
        self.batch_times = []  # รายการสำหรับเก็บเวลาแบตช์
        self.forward_times = []  # รายการสำหรับเก็บเวลา forward pass
        self.backward_times = []  # รายการสำหรับเก็บเวลา backward pass
        self.gpu_memory_usage = []  # รายการสำหรับเก็บการใช้งานหน่วยความจำ GPU
        self.cpu_memory_usage = []  # รายการสำหรับเก็บการใช้งานหน่วยความจำ CPU

    def calculate_averages(self):
        return {
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,  # คำนวณค่าเฉลี่ยเวลาแบตช์
            'avg_forward_time': np.mean(self.forward_times) if self.forward_times else 0,  # คำนวณค่าเฉลี่ยเวลา forward
            'avg_backward_time': np.mean(self.backward_times) if self.backward_times else 0,  # คำนวณค่าเฉลี่ยเวลา backward
            'avg_gpu_memory': np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,  # คำนวณค่าเฉลี่ยการใช้งาน GPU
            'avg_cpu_memory': np.mean(self.cpu_memory_usage) if self.cpu_memory_usage else 0  # คำนวณค่าเฉลี่ยการใช้งาน CPU
        }

    def print_summary(self):
        averages = self.calculate_averages()  # คำนวณค่าเฉลี่ย
        print("\nBenchmark Summary:")  # แสดงหัวข้อ
        print("=" * 50)  # เส้นแบ่ง
        print(f"Average Batch Processing Time: {averages['avg_batch_time']:.4f} seconds")  # แสดงค่าเฉลี่ยเวลาแบตช์
        print(f"Average Forward Pass Time: {averages['avg_forward_time']:.4f} seconds")  # แสดงค่าเฉลี่ยเวลา forward
        print(f"Average Backward Pass Time: {averages['avg_backward_time']:.4f} seconds")  # แสดงค่าเฉลี่ยเวลา backward
        print(f"Average GPU Memory Usage: {averages['avg_gpu_memory']:.2f} GB")  # แสดงค่าเฉลี่ยการใช้งาน GPU
        print(f"Average CPU Memory Usage: {averages['avg_cpu_memory']:.2f} GB")  # แสดงค่าเฉลี่ยการใช้งาน CPU
        print("=" * 50)  # เส้นแบ่ง


def train_model():  # ฟังก์ชันสำหรับการฝึกโมเดล
    print(f"\n{'=' * 50}")  # แสดงเส้นแบ่ง
    print(f"CNN-LSTM Training")  # แสดงหัวข้อการฝึก
    print(f"{'=' * 50}")  # แสดงเส้นแบ่ง

    # กำหนดค่าพารามิเตอร์ที่แน่นอน
    num_epochs = 200  # จำนวนยุคในการฝึก
    dropout = 0.1  # ค่าดรอปเอาต์
    learning_rate = 0.001  # อัตราการเรียนรู้
    batch_size = 37  # ขนาดของแบตช์
    hidden_units = 150  # จำนวนยูนิตใน LSTM
    hidden_layers = 3  # จำนวนชั้นใน LSTM

    # สร้าง Visualizer และ Benchmark Metrics
    visualizer = TrainingVisualizer(save_dir=plot, dropout=dropout)  # สร้างอ็อบเจ็กต์สำหรับการแสดงผล
    benchmark = BenchmarkMetrics()  # สร้างอ็อบเจ็กต์สำหรับค่าการวัดประสิทธิภาพ

    # แสดงพารามิเตอร์
    print("\nParameters:")
    print(f"-- Batch Size: {batch_size}")  # แสดงขนาดของแบตช์
    print(f"-- Hidden Units: {hidden_units}")  # แสดงจำนวนยูนิตใน LSTM
    print(f"-- Hidden Layers: {hidden_layers}")  # แสดงจำนวนชั้นใน LSTM
    print(f"-- Number of Epochs: {num_epochs}")  # แสดงจำนวนยุคในการฝึก
    print(f"-- Learning Rate: {learning_rate:.3f}")  # แสดงอัตราการเรียนรู้
    print(f"-- Dropout: {dropout:.1f}")  # แสดงค่าดรอปเอาต์

    print(f"{'=' * 50}")  # แสดงเส้นแบ่ง

   # ดึงข้อมูลการฝึก, การตรวจสอบ และการทดสอบ
    X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.get_train_test_data()  # ดึงข้อมูลจาก VideoPreprocessor

    # รวมชุดข้อมูลการฝึกและการตรวจสอบ
    train_data = np.concatenate([X_train, X_val])  # รวมข้อมูลการฝึก
    train_labels = np.concatenate([y_train, y_val])  # รวม labels

    num_sequences = 1  # กำหนดจำนวน sequences
    train_dataset = HandGestureDataset(train_data, train_labels, num_sequences=num_sequences)  # สร้าง HandGestureDataset สำหรับการฝึก
    test_dataset = HandGestureDataset(X_test, y_test, num_sequences=num_sequences)  # สร้าง HandGestureDataset สำหรับการทดสอบ

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # สร้าง DataLoader สำหรับการฝึก
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # สร้าง DataLoader สำหรับการทดสอบ

    model = CNN_LSTM(  # สร้างโมเดล CNN-LSTM
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        dropout=dropout
    ).to(device)  # ส่งโมเดลไปยังอุปกรณ์

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # กำหนดออปติไมเซอร์ Adam
    criterion = nn.CrossEntropyLoss()  # กำหนดฟังก์ชันสูญเสีย CrossEntropy

    # วนลูปการฝึกอบรม
    best_val_acc = 0.0  # ตัวแปรสำหรับเก็บ accuracy ที่ดีที่สุด
    print("\nTraining Progress:")  # แสดงหัวข้อการฝึก
    print("-" * 50)  # แสดงเส้นแบ่ง

    for epoch in range(num_epochs):  # วนลูปตามจำนวนยุค
        model.train()  # ตั้งค่าโมเดลเป็นโหมดการฝึก
        train_correct = 0  # ตัวแปรสำหรับเก็บจำนวนการทำนายที่ถูกต้องในการฝึก
        train_total = 0  # ตัวแปรสำหรับเก็บจำนวนข้อมูลทั้งหมดในการฝึก
        train_losses = []  # รายการสำหรับเก็บค่า loss ของการฝึก
        epoch_start_time = time.time()  # เริ่มต้นนับเวลาในการฝึกแต่ละยุค
        batch_times_epoch = []  # รายการสำหรับเก็บเวลาในแต่ละแบตช์
        gpu_memory_epoch = []  # รายการสำหรับเก็บการใช้งานหน่วยความจำ GPU ในแต่ละยุค

        # เฟสการฝึกพร้อมการวัดประสิทธิภาพ
        for batch_idx, (sequences, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):  # วนลูปผ่าน DataLoader
            batch_start_time = time.time()  # เริ่มนับเวลาในการประมวลผลแบตช์

            sequences, labels = sequences.to(device), labels.to(device)  # ส่งข้อมูลและ labels ไปยังอุปกรณ์
            
            # ล้าง gradients
            optimizer.zero_grad()  # ล้างค่าของ gradients

            # การวัดเวลา forward pass
            forward_start = time.time()  # เริ่มนับเวลา
            outputs = model(sequences)  # ประมวลผลข้อมูลผ่านโมเดล
            forward_time = time.time() - forward_start  # คำนวณเวลา forward
            benchmark.forward_times.append(forward_time)  # เก็บเวลา forward

            # คำนวณ loss และวัดเวลา backward pass
            loss = criterion(outputs, labels)  # คำนวณ loss
            backward_start = time.time()  # เริ่มนับเวลา
            loss.backward()  # คำนวณ gradients
            optimizer.step()  # ปรับปรุงน้ำหนัก
            backward_time = time.time() - backward_start  # คำนวณเวลา backward
            benchmark.backward_times.append(backward_time)  # เก็บเวลา backward

            # เก็บเมตริก
            train_losses.append(loss.item())  # เพิ่ม loss ในรายการ
            _, predicted = torch.max(outputs.data, 1)  # ทำนายคลาสจากผลลัพธ์
            train_total += labels.size(0)  # เพิ่มจำนวนข้อมูลทั้งหมด
            train_correct += (predicted == labels).sum().item()  # คำนวณจำนวนการทำนายที่ถูกต้อง

            # วัดเวลาในการประมวลผลแบตช์
            batch_time = time.time() - batch_start_time  # คำนวณเวลาแบตช์
            batch_times_epoch.append(batch_time)  # เก็บเวลาแบตช์
            benchmark.batch_times.append(batch_time)  # เก็บเวลาแบตช์ใน Benchmark

            # ติดตามการใช้งานหน่วยความจำ
            if torch.cuda.is_available():  # ถ้าใช้ GPU
                gpu_memory = torch.cuda.memory_allocated() / 1e9  # แปลงเป็น GB
                gpu_memory_epoch.append(gpu_memory)  # เก็บการใช้งาน GPU
                benchmark.gpu_memory_usage.append(gpu_memory)  # เก็บการใช้งาน GPU ใน Benchmark

            # ติดตามการใช้งานหน่วยความจำ CPU
            process = psutil.Process()  # สร้างอ็อบเจ็กต์สำหรับติดตามการใช้งาน CPU
            cpu_memory = process.memory_info().rss / 1e9  # แปลงเป็น GB
            benchmark.cpu_memory_usage.append(cpu_memory)  # เก็บการใช้งาน CPU ใน Benchmark

            # ล้างแคชเป็นระยะ
            if batch_idx % 10 == 0:  # ทุกๆ 10 แบตช์
                torch.cuda.empty_cache()  # ล้างแคชของ GPU

        train_acc = 100 * train_correct / train_total  # คำนวณ accuracy ของการฝึก
        train_loss = np.mean(train_losses)  # คำนวณค่าเฉลี่ย loss ของการฝึก
        epoch_time = time.time() - epoch_start_time  # คำนวณเวลาในการฝึกแต่ละยุค

        # เฟสการตรวจสอบพร้อมเมตริก
        model.eval()  # ตั้งค่าโมเดลเป็นโหมดตรวจสอบ
        val_correct = 0  # ตัวแปรสำหรับเก็บจำนวนการทำนายที่ถูกต้องในการตรวจสอบ
        val_total = 0  # ตัวแปรสำหรับเก็บจำนวนข้อมูลทั้งหมดในการตรวจสอบ
        val_losses = []  # รายการสำหรับเก็บค่า loss ของการตรวจสอบ
        val_predictions = []  # รายการสำหรับเก็บการทำนายของการตรวจสอบ
        val_true_labels = []  # รายการสำหรับเก็บ label จริงของการตรวจสอบ

        with torch.no_grad():  # ปิดการคำนวณ gradients
            for sequences, labels in test_loader:  # วนลูปผ่าน DataLoader
                sequences, labels = sequences.to(device), labels.to(device)  # ส่งข้อมูลไปยังอุปกรณ์

                forward_start = time.time()  # เริ่มนับเวลา
                outputs = model(sequences)  # ประมวลผลข้อมูลผ่านโมเดล
                forward_time = time.time() - forward_start  # คำนวณเวลา forward
                benchmark.forward_times.append(forward_time)  # เก็บเวลา forward

                loss = criterion(outputs, labels)  # คำนวณ loss
                val_losses.append(loss.item())  # เพิ่ม loss ในรายการ

                _, predicted = torch.max(outputs.data, 1)  # ทำนายคลาสจากผลลัพธ์
                val_predictions.extend(predicted.cpu().numpy())  # เก็บการทำนายในรายการ
                val_true_labels.extend(labels.cpu().numpy())  # เก็บ label จริงในรายการ
                val_total += labels.size(0)  # เพิ่มจำนวนข้อมูลทั้งหมด
                val_correct += (predicted == labels).sum().item()  # คำนวณจำนวนการทำนายที่ถูกต้อง

        # คำนวณเมตริก
        val_acc = 100 * val_correct / val_total  # คำนวณ accuracy ของการตรวจสอบ
        val_loss = np.mean(val_losses)  # คำนวณค่าเฉลี่ย loss ของการตรวจสอบ
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')  # คำนวณ F1 Score
        val_precision = precision_score(val_true_labels, val_predictions, average='weighted')  # คำนวณ Precision
        val_recall = recall_score(val_true_labels, val_predictions, average='weighted')  # คำนวณ Recall

        # อัปเดตเมตริกเพื่อการแสดงผล
        epoch_metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gpu_memory': np.mean(gpu_memory_epoch) if gpu_memory_epoch else 0,
            'batch_times': np.mean(batch_times_epoch),
            'epoch_times': epoch_time
        }
        visualizer.update_metrics(epoch_metrics)  # อัปเดตเมตริกใน visualizer

        # บันทึกกราฟหลังจากแต่ละยุค
        visualizer.plot_all_metrics()  # สร้างกราฟทั้งหมด

        # ติดตาม accuracy ที่ดีที่สุดในการตรวจสอบ
        if val_acc > best_val_acc:  # ถ้า accuracy ใหม่สูงกว่า
            best_val_acc = val_acc  # อัปเดตค่า accuracy ที่ดีที่สุด

       # แสดงผลลัพธ์ของยุคพร้อมข้อมูลการวัดประสิทธิภาพ
        print(f"\nEpoch {epoch + 1}")  # แสดงหมายเลขยุค
        print(f"Training Metrics:")  # แสดงหัวข้อเมตริกการฝึก
        print(f"- Accuracy: {train_acc:.2f}%")  # แสดง accuracy ของการฝึก
        print(f"- Loss: {train_loss:.4f}")  # แสดง loss ของการฝึก
        print(f"Validation Metrics:")  # แสดงหัวข้อเมตริกการตรวจสอบ
        print(f"- Accuracy: {val_acc:.2f}%")  # แสดง accuracy ของการตรวจสอบ
        print(f"- F1 Score: {val_f1:.4f}")  # แสดง F1 Score
        print(f"- Precision: {val_precision:.4f}")  # แสดง Precision
        print(f"- Recall: {val_recall:.4f}")  # แสดง Recall

        # แสดงข้อมูลการวัดประสิทธิภาพตามแบตช์ปัจจุบัน
        batch_averages = benchmark.calculate_averages()  # คำนวณค่าเฉลี่ย
        print("\nCurrent Benchmarks:")  # แสดงหัวข้อการวัดประสิทธิภาพ
        print(f"- Avg Batch Time: {batch_averages['avg_batch_time']:.4f} seconds")  # แสดงค่าเฉลี่ยเวลาแบตช์
        print(f"- Avg Forward Time: {batch_averages['avg_forward_time']:.4f} seconds")  # แสดงค่าเฉลี่ยเวลา forward
        print(f"- Avg Backward Time: {batch_averages['avg_backward_time']:.4f} seconds")  # แสดงค่าเฉลี่ยเวลา backward
        print(f"- Current GPU Memory: {batch_averages['avg_gpu_memory']:.2f} GB")  # แสดงการใช้งาน GPU ปัจจุบัน
        print("-" * 30)  # แสดงเส้นแบ่ง

    # แสดงสรุปการวัดประสิทธิภาพสุดท้าย
    benchmark.print_summary()  # แสดงสรุปการวัดประสิทธิภาพ

    # สรุปผลการฝึกอบรม
    print(f"\nTraining Summary:")  # แสดงหัวข้อสรุปการฝึก
    print(f"{'=' * 30}")  # แสดงเส้นแบ่ง
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")  # แสดง accuracy ที่ดีที่สุดในการตรวจสอบ
    print(f"Final Validation Metrics:")  # แสดงหัวข้อเมตริกการตรวจสอบสุดท้าย
    print(f"- F1 Score: {val_f1:.4f}")  # แสดง F1 Score สุดท้าย
    print(f"- Precision: {val_precision:.4f}")  # แสดง Precision สุดท้าย
    print(f"- Recall: {val_recall:.4f}")  # แสดง Recall สุดท้าย
    print(f"Parameters:")  # แสดงหัวข้อพารามิเตอร์
    print(f"-- Batch Size: {batch_size}")  # แสดงขนาดของแบตช์
    print(f"-- Hidden Units: {hidden_units}")  # แสดงจำนวนยูนิตใน LSTM
    print(f"-- Hidden Layers: {hidden_layers}")  # แสดงจำนวนชั้นใน LSTM
    print(f"-- Number of Epochs: {num_epochs}")  # แสดงจำนวนยุคในการฝึก
    print(f"-- Learning Rate: {learning_rate:.6f}")  # แสดงอัตราการเรียนรู้
    print(f"-- Dropout: {dropout:.6f}")  # แสดงค่าดรอปเอาต์
    print(f"{'=' * 50}")  # แสดงเส้นแบ่ง

    return model, best_val_acc, test_loader  # คืนค่าโมเดล, accuracy ที่ดีที่สุด และ DataLoader สำหรับการทดสอบ


def evaluate_final_results(model, test_loader, device):  # ฟังก์ชันสำหรับประเมินผลลัพธ์สุดท้าย
    print(f"\n{'=' * 50}")  # แสดงเส้นแบ่ง
    print(f"Final Model Evaluation")  # แสดงหัวข้อการประเมินผลโมเดลสุดท้าย
    print(f"{'=' * 50}")  # แสดงเส้นแบ่ง

    model.eval()  # ตั้งค่าโมเดลเป็นโหมดตรวจสอบ
    test_pred = []  # รายการสำหรับเก็บการทำนายของชุดทดสอบ
    test_true = []  # รายการสำหรับเก็บ label จริงของชุดทดสอบ

    # สร้างการทำนาย
    print("\nGenerating predictions for test set...")  # แจ้งกำลังสร้างการทำนาย
    with torch.no_grad():  # ปิดการคำนวณ gradients
        for sequences, labels in tqdm(test_loader, desc="Evaluating"):  # วนลูปผ่าน DataLoader
            sequences = sequences.to(device)  # ส่งข้อมูลไปยังอุปกรณ์
            outputs = model(sequences)  # ประมวลผลข้อมูลผ่านโมเดล
            predictions = torch.argmax(outputs, dim=1)  # ทำนายคลาสจากผลลัพธ์
            test_pred.extend(predictions.cpu().numpy())  # เก็บการทำนายในรายการ
            test_true.extend(labels.cpu().numpy())  # เก็บ label จริงในรายการ

    # คำนวณเมตริก
    accuracy = accuracy_score(test_true, test_pred) * 100  # คำนวณ accuracy
    f1 = f1_score(test_true, test_pred, average='weighted')  # คำนวณ F1 Score
    precision = precision_score(test_true, test_pred, average='weighted')  # คำนวณ Precision
    recall = recall_score(test_true, test_pred, average='weighted')  # คำนวณ Recall

    # แสดง confusion matrix
    print("\nConfusion Matrix:")  # แสดงหัวข้อ confusion matrix
    print("-" * 50)  # แสดงเส้นแบ่ง
    class_names = ["No_Requests", "Person_1_Requests",  # ชื่อคลาส
                   "Person_2_Requests", "Person_1_and_Person_2_Requests"]
    cm = confusion_matrix(test_true, test_pred)  # คำนวณ confusion matrix

    print("\nClass-wise Results:")  # แสดงหัวข้อผลลัพธ์ตามคลาส
    print("-" * 50)  # แสดงเส้นแบ่ง
    print(classification_report(test_true, test_pred, target_names=class_names))  # แสดงรายงานการจำแนกประเภท

    # แสดงเมตริกสุดท้าย
    print("\nFINAL TEST SET RESULTS")  # แสดงหัวข้อผลลัพธ์ชุดทดสอบสุดท้าย
    print("=" * 50)  # แสดงเส้นแบ่ง
    print(f"Test Accuracy: {accuracy:.2f}%")  # แสดง accuracy ของชุดทดสอบ
    print(f"Test F1 Score: {f1:.4f}")  # แสดง F1 Score ของชุดทดสอบ
    print(f"Test Precision: {precision:.4f}")  # แสดง Precision ของชุดทดสอบ
    print(f"Test Recall: {recall:.4f}")  # แสดง Recall ของชุดทดสอบ
    print("=" * 50)  # แสดงเส้นแบ่ง

    # วาด confusion matrix
    plt.figure(figsize=(10, 8))  # กำหนดขนาดของกราฟ
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # สร้าง heatmap สำหรับ confusion matrix
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')  # ตั้งชื่อกราฟ
    plt.ylabel('True Label')  # ตั้งชื่อแกน y
    plt.xlabel('Predicted Label')  # ตั้งชื่อแกน x
    plt.xticks(rotation=45)  # หมุนชื่อแกน x
    plt.yticks(rotation=45)  # หมุนชื่อแกน y
    plt.tight_layout()  # ปรับขนาดกราฟให้เหมาะสม

    # บันทึก confusion matrix ในโฟลเดอร์ที่กำหนด (D:\cnn_lstm\Plot)
    confusion_matrix_path = os.path.join(plot, f'confusion_matrix.png')  # ตั้งชื่อไฟล์
    plt.savefig(confusion_matrix_path)  # บันทึกกราฟ
    plt.close()  # ปิดกราฟ
    print(f"\nConfusion matrix saved to: {confusion_matrix_path}")  # แสดงข้อความยืนยันการบันทึก


def main():  # ฟังก์ชันหลัก
    # ข้อมูลเกี่ยวกับ GPU
    print(f"Using device: {device}")  # แสดงอุปกรณ์ที่ใช้
    print(f"CUDA available: {torch.cuda.is_available()}")  # แจ้งว่ามี CUDA หรือไม่
    if torch.cuda.is_available():  # ถ้ามี CUDA
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")  # แสดงชื่อของ GPU

    # ฝึกโมเดล
    model, best_acc, test_loader = train_model()  # เรียกใช้ฟังก์ชันฝึกโมเดล
    print(f"Training completed with best accuracy: {best_acc:.2f}%")  # แสดงผลการฝึกที่ดีที่สุด

    # ประเมินผลลัพธ์สุดท้าย
    evaluate_final_results(model, test_loader, device)  # เรียกใช้ฟังก์ชันประเมินผล


if __name__ == "__main__":  # ถ้ารันไฟล์นี้เป็นโปรแกรมหลัก
    main()  # เรียกใช้ฟังก์ชันหลัก
