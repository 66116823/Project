import random  # นำเข้าโมดูล random สำหรับการสุ่ม
import cv2 # นำเข้า OpenCV สำหรับการประมวลผลภาพ
import numpy as np # นำเข้า NumPy สำหรับการคำนวณเชิงตัวเลข
from pathlib import Path # นำเข้า Path สำหรับการจัดการไฟล์
import torch # นำเข้า PyTorch สำหรับการสร้างโมเดลและการคำนวณ
import torch.nn as nn # นำเข้าโมดูล neural network จาก PyTorch
from torch.utils.data import Dataset, DataLoader # นำเข้า Dataset และ DataLoader สำหรับการจัดการข้อมูล
import torchvision.transforms as transforms # นำเข้า transforms สำหรับการแปลงภาพ
from sklearn.metrics import classification_report, confusion_matrix # นำเข้าเครื่องมือสำหรับวัดประสิทธิภาพ

from cnn_preprocessing import load_video # นำเข้าฟังก์ชันสำหรับโหลดวิดีโอ
from cnn_preprocessing import select_frames # นำเข้าฟังก์ชันสำหรับเลือกเฟรม
from cnn_preprocessing import split_dataset # นำเข้าฟังก์ชันสำหรับแบ่งชุดข้อมูล
from cnn_preprocessing import folder_paths,load_videos_from_folders # นำเข้าที่อยู่ของโฟลเดอร์และโหลดวิดีโอ
from PIL import Image # นำเข้า PIL สำหรับการจัดการภาพ
from tqdm import tqdm # นำเข้า tqdm สำหรับแสดงแถบความก้าวหน้า


# Parameters # กำหนดจำนวนคลาส
num_classes = 4

# สร้างคลาส HandGestureDataset สำหรับจัดการชุดข้อมูล
class HandGestureDataset(Dataset):
    def __init__(self, video_paths, labels, sequence_length=15, transform=None):
        self.video_paths = video_paths # ที่อยู่ของวิดีโอ
        self.labels = labels # ป้ายกำกับของวิดีโอ
        self.sequence_length = sequence_length # ความยาวของลำดับเฟรม
        self.transform = transform # การแปลงที่ใช้กับเฟรม

    def __len__(self): # เมธอดสำหรับคืนค่าจำนวนตัวอย่างในชุดข้อมูล
        return len(self.video_paths)
    
    def __getitem__(self, idx): # เมธอดสำหรับเข้าถึงตัวอย่างในชุดข้อมูล
        try:
            video_path = self.video_paths[idx] # ดึงที่อยู่ของวิดีโอ

            # ตรวจสอบว่ามีไฟล์วิดีโออยู่หรือไม่
            if not Path(video_path).is_file():
                print(f"Warning: Video file not found - {video_path}")
                # ส่งคืนค่าเริ่มต้นหรือข้ามตัวอย่างนี้
                return self.__getitem__((idx + 1) % len(self.video_paths))
            
            # โหลดวิดีโอและเลือกเฟรม บน GPU
            frames = load_video(video_path)

            if frames.shape[0] == 0:  # ถ้า frames เป็นเทนเซอร์ว่าง
                print(f"Warning: No frames loaded from video - {video_path}")
                return self.__getitem__((idx + 1) % len(self.video_paths))

            # เลือกเฟรมบน GPU
            frames = select_frames(frames, self.sequence_length)

            # ตรวจสอบกรณีที่ไม่มีเฟรมหรือเฟรมว่าง เพื่อประมวลผลเฟรม
            if frames is None or len(frames) == 0:
                # กลยุทธ์สำรอง: พยายามโหลดวิดีโออื่นหรือส่งค่าเริ่มต้น
                # ในกรณีนี้ เราจะส่งข้อผิดพลาดที่มีข้อมูลมากขึ้น
                print(f"Warning: No frames loaded from video - {video_path}")
                return self.__getitem__((idx + 1) % len(self.video_paths))
            
            
            
            # # แปลงเป็น numpy array ถ้ายังไม่ใช่
            # frames = np.array(frames)
                
            # # ตรวจสอบมิติของเฟรม
            # if frames.ndim < 3: # น้อยกว่า 3 ไม่ได้  = ข้อมูลไม่สมบูรณ์
            #     raise ValueError(f"เลือกเฟรมจากวิดีโอไม่เพียงพอ: {self.video_paths[idx]}")
                
            # # ตรวจสอบความยาวของลำดับเฟรม
            # if frames.shape[0] < self.sequence_length:
            #     # เติมเฟรมถ้าเลือกได้ไม่ครบ
            #     pad_width = ((0, self.sequence_length - frames.shape[0]), (0, 0), (0, 0), (0, 0))
            #     frames = np.pad(frames, pad_width, mode='constant')
            # elif frames.shape[0] > self.sequence_length:
            #     # ตัดทอนถ้ามีเฟรมมากเกินไป
            #     frames = frames[:self.sequence_length]
                
            # # ตรวจสอบขนาดมิติของเฟรม
            # if len(frames.shape) != 4:  # (sequence_length, height, width, channels)
            #     raise ValueError(f"รูปร่างเฟรมไม่เป็นไปตามที่คาดหวัง: {frames.shape}")
                
            # แปลงและประมวลผลเฟรม
            processed_frames = []
            for i in range(self.sequence_length):
                frame = frames[i]  # (height, width, channels)
                    
                # แปลงเป็น PIL Image 
                frame_pil = transforms.ToPILImage()(frame.cpu())
                    
                # ใช้การแปลงถ้ามี
                if self.transform:
                    frame = self.transform(frame_pil)
                    
                processed_frames.append(frame)
            
            # รวมเฟรมเป็นเทนเซอร์เดียว
            frames_tensor = torch.stack(processed_frames)
            
            # # ตรวจสอบรูปร่างสุดท้ายของเทนเซอร์
            # expected_shape = (self.sequence_length, 3, 128, 128)  # ปรับตามความต้องการ
            # if frames_tensor.shape != expected_shape:
            #     raise ValueError(f"รูปร่างเทนเซอร์ที่คาดหวัง {expected_shape}, ได้รับ {frames_tensor.shape}")
                
            # แปลงป้ายกำกับเป็นเทนเซอร์
            label = torch.tensor(self.labels[idx], dtype=torch.long)
                
            return frames_tensor, label
        
        except Exception as e:
            print(f"Error processing video {self.video_paths[idx]}: {e}")
            # Fallback to another sample if processing fails
            return self.__getitem__((idx + 1) % len(self.video_paths))

    

# CNN module
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            #### Conv Layer 1
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            # in_channels กำหนด เป็น 3 เพราะเป็นภาพสี RGB(Red, Green, Blue) ของพี่แจกันเป็น 1 ภาพขาวดำ

            # Input size: 128*128*3 (W1,H1, D1(Depth) = จำนวนช่องสัญญาณ)
            # Spatial extend of each one (kernelConv size), F = 3
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((128 - 3 + 2 * 2) / 1) + 1 = 130 #*# W2 = (( W1 - F + 2(P) ) / S ) + 1
            ## High: ((128 - 3 + 2 * 2) / 1) + 1 = 130 #*# H2 = (( H1 - F + 2(P) ) / S ) + 1
            ## Depth: 32
            ## Output Conv Layer1: 130 * 130 * 32 
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),

            ### Max Pooling Layer 1 
            # Input size: 130 * 130 * 32 (เอามาจาก output Conv layer1)
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 1
            ## Width: ((130 - 2) / 2) + 1 = 65 #*# (( W2 - F ) / S ) + 1
            ## High: ((130 - 2) / 2) + 1 = 65 #*# (( H2 - F ) / S ) + 1
            ## Depth: 32
            ### Output Max Pooling Layer 1: 65 * 65 * 32 ผ่าน Max Pooling ขนาดจะลดลงครึ่งนึง
            # (kernel size pooling(2), stride(2)) 
            nn.MaxPool2d(2, 2), 
            
            
            #### Conv Layer 2
            # in_channels ใน layer นี้มาจากoutput(layerแรก) = 32
            # Input size: 65*65*32 (W1,H1, D1(Depth) = จำนวนช่องสัญญาณ)
            # Spatial extend of each one (kernelConv size), F = 3
            # Slide size (strideConv), S = 1
            # Padding, P = 2
            ## Width: ((65 - 3 + 2 * 2) / 1) + 1 = 67 #*# W3 = (( W2 - F + 2(P) ) / S ) + 1
            ## High: ((65 - 3 + 2 * 2) / 1) + 1 =  67#*# H3 = (( H2 - F + 2(P) ) / S ) + 1
            ## Depth: 64
            ## Output Conv Layer2: 67 * 67 * 64 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),

            ### Max Pooling Layer 2
            # Input size: 67 * 67 * 64 (เอามาจาก output Conv layer2)
            ## Spatial extend of each one (kernelMaxPool size), F = 2
            ## Slide size (strideMaxPool), S = 2
            # Output Max Pooling Layer 2
            ## Width: ((67 - 2) / 2) + 1 = 33.5 #*# (( W3 - F ) / S ) + 1
            ## High: ((67 - 2) / 2) + 1 = 33.5 #*# (( H3 - F ) / S ) + 1
            ## Depth: 64
            ### Output Max Pooling Layer 1: 33.5 * 33.5 * 64 ผ่าน Max Pooling ขนาดจะลดลงครึ่งนึง

            # (kernel size pooling(2), stride(2)) 
            nn.MaxPool2d(2, 2), 
            
        
            # nn.AdaptiveAvgPool2d((4, 4))  # ปรับขนาด output ให้คงที่เป็น 4*4 ต่อให้จะกำหนดความกว้างยาว input=128*128
        )
        
    def forward(self, x):
        return self.features(x)

# CNN-LSTM Model
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN()
        self.lstm = nn.LSTM(
            input_size= 33 * 33 * 64,  # เอามาจาก CNN ที่ผ่านมา 2 Layer
            # hidden_size = 100 (ตามพี่แจกัน)
            hidden_size= 100, # เป็นการกำหนดจำนวนของ node ในชั้น hidden layer ถ้าเยอะเกินจะ overfit น้อยเกินโมเดลเรียนรู้ไม่เพียงพอ
            # num layers = 1 (ตามพี่แจกัน)
            num_layers=1, 
            batch_first=True)
        
        # self.fc = nn.Sequential(
        #         nn.Linear(256, 128),
        #         nn.ReLU(),
        #         # nn.Dropout(0.5),
        #         nn.Linear(100, num_classes)
        #     )
        ## ตามพี่แจกัน
        self.fc = nn.Linear(100, num_classes) # (hidden_size, numclasses)
            

    def forward(self, x):
        # x มีขนาด [batch_size, sequence_length=15, channels=3, height, width]
        batch_size, seq_len, C, H, W = x.size()

        # จัดรูปข้อมูลให้ CNN ประมวลผลทีละเฟรม
        # รวม batch และ sequence dimensions สำหรับ CNN
        x = x.view(batch_size * seq_len, C, H, W)
        # ประมวลผลทั้ง 15 เฟรมผ่าน CNN
        x = self.cnn(x) 

        # จัดรูปข้อมูลกลับเป็น sequence สำหรับ LSTM
        x = x.view(batch_size, seq_len, -1)
        # LSTM จะดู features ของทั้ง 15 เฟรมตามลำดับเวลา
        lstm_out, _ = self.lstm(x)
        # output = เป็น hidden state สุดท้ายที่ lstm เรียนรู้จากการดูทั้ง 15 เฟรม
        x = self.fc(lstm_out[:, -1, :]) 

        return x # คืนค่าผลลัพธ์


def train_model(model, train_loader, val_loader, test_loader, num_epochs=10, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # ตรวจสอบว่าใช้ GPU หรือไม่
    model = model.to(device) # ส่งโมเดลไปยังอุปกรณ์ที่เลือก

    # การฝึกอบรมแบบ mixed precision
    scaler = torch.amp.GradScaler() # สร้าง scaler สำหรับการจัดการ gradient

    criterion = nn.CrossEntropyLoss() # กำหนด loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5) # สร้าง optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) # สร้าง scheduler สำหรับปรับ learning rate
    
    best_acc = 0.0 # ตัวแปรสำหรับเก็บความแม่นยำที่ดีที่สุด
    no_improve = 0 # ตัวนับสำหรับการไม่มีการปรับปรุง

    for epoch in range(num_epochs): # วนลูปตามจำนวน epoch
        # Training
        model.train() # เปลี่ยนโมเดลเป็นโหมดฝึกอบรม
        train_loss = 0.0 # ตัวแปรสำหรับเก็บค่า loss ของการฝึก
        train_correct = 0 # ตัวแปรสำหรับเก็บจำนวนการทำนายที่ถูกต้องในการฝึก
        train_total = 0 # ตัวแปรสำหรับเก็บจำนวนตัวอย่างทั้งหมดในการฝึก

        for i, (sequences, labels) in enumerate(train_loader):  # วนลูปผ่าน data loader
            sequences, labels = sequences.to(device), labels.to(device)   # ส่งข้อมูลไปยังอุปกรณ์ที่เลือก

             # ทำการประมวลผลด้วย autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # ใช้ mixed precision
                outputs = model(sequences) # ส่งข้อมูลผ่านโมเดล
                loss = criterion(outputs, labels)  # คำนวณค่า loss

            # Scales loss and calls backward() to create scaled gradients  # สร้าง gradient
            scaler.scale(loss).backward() # สร้าง gradient ด้วยการ scale loss
            
            # Unscales gradients and calls optimizer.step()  # อัปเดตโมเดล
            scaler.step(optimizer)  # อัปเดตน้ำหนักของโมเดล

            # Updates the scale for next iteration # อัปเดต scale สำหรับการวนรอบถัดไป
            scaler.update()

            optimizer.zero_grad()  # รีเซ็ต gradient

            # outputs = model(sequences)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            train_loss += loss.item() # เพิ่มค่า loss เข้าไปในตัวแปรรวม
            _, predicted = torch.max(outputs.data, 1) # คำนวณการทำนาย
            train_total += labels.size(0) # เพิ่มจำนวนตัวอย่างทั้งหมด
            train_correct += (predicted == labels).sum().item() # เพิ่มจำนวนการทำนายที่ถูกต้อง
            
            if (i+1) % 10 == 0: # แสดงค่าทุก ๆ 10 step
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_correct = 0 # ตัวแปรสำหรับเก็บจำนวนการทำนายที่ถูกต้องในการทดสอบ
        val_total = 0 # ตัวแปรสำหรับเก็บจำนวนตัวอย่างทั้งหมดในการทดสอบ
        val_loss = 0.0 # ตัวแปรสำหรับเก็บค่า loss ของการทดสอบ
         
        with torch.no_grad(): # ปิดการคำนวณ gradient
            for sequences, labels in val_loader: # วนลูปผ่าน validation data loader
                sequences, labels = sequences.to(device), labels.to(device) # ส่งข้อมูลไปยังอุปกรณ์ที่เลือก
                outputs = model(sequences) # ส่งข้อมูลผ่านโมเดล
                val_loss += criterion(outputs, labels).item() # คำนวณค่า loss สำหรับ validation
                _, predicted = torch.max(outputs.data, 1) # คำนวณการทำนาย
                val_total += labels.size(0) # เพิ่มจำนวนตัวอย่างทั้งหมด
                val_correct += (predicted == labels).sum().item() # เพิ่มจำนวนการทำนายที่ถูกต้อง

        # Testing
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad(): # ปิดการคำนวณ gradient
            for sequences, labels in test_loader: # วนลูปผ่าน test data loader
                sequences, labels = sequences.to(device), labels.to(device) # ส่งข้อมูลไปยังอุปกรณ์ที่เลือก
                outputs = model(sequences) # ส่งข้อมูลผ่านโมเดล
                test_loss += criterion(outputs, labels).item() # คำนวณค่า loss สำหรับ test
                _, predicted = torch.max(outputs.data, 1) # คำนวณการทำนาย
                test_total += labels.size(0) # เพิ่มจำนวนตัวอย่างทั้งหมด
                test_correct += (predicted == labels).sum().item() # เพิ่มจำนวนการทำนายที่ถูกต้อง

        # Calculate accuracies # คำนวณความแม่นยำ
        train_acc = 100 * train_correct / train_total # คำนวณความแม่นยำในการฝึก
        val_acc = 100 * val_correct / val_total # คำนวณความแม่นยำในการ validation
        test_acc = 100 * test_correct / test_total # คำนวณความแม่นยำในการทดสอบ
        
        # Print epoch results # แสดงผลลัพธ์ของ epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%')

        # Update best accuracy based on validation performance # อัปเดตความแม่นยำที่ดีที่สุด
        if val_acc > best_acc:
            best_acc = val_acc # อัปเดตความแม่นยำที่ดีที่สุด
            no_improve = 0 # รีเซ็ตตัวนับ
        else:
            no_improve += 1 # เพิ่มตัวนับถ้าไม่มีการปรับปรุง

        # Learning rate scheduling based on validation loss # ปรับ learning rate ตาม validation loss
        scheduler.step(val_loss)

        # Early stopping # ตรวจสอบการหยุดการฝึก
        if no_improve >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break

    return model, best_acc # คืนค่าโมเดลและความแม่นยำที่ดีที่สุด
# ฟังก์ชันสำหรับทดลองขนาด batch
def experiment_batch_sizes(batch_sizes=[8, 16, 32, 64], num_runs=3):
    # Use mixed precision training
    scaler = torch.amp.GradScaler() # สร้าง scaler สำหรับการจัดการ gradient

    # Transformations # กำหนดการแปลงภาพ
    transform = transforms.Compose([
        transforms.ToTensor(), # แปลงภาพเป็น tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ปรับค่าความเข้มและการกระจาย
        # Optional: data augmentation for better generalization
        transforms.RandomHorizontalFlip(p=0.5) # การเพิ่มข้อมูลโดยการพลิกภาพ
    ])

    # Use more efficient data loading # ฟังก์ชันสำหรับกำหนด seed สำหรับ worker
    def seed_worker(worker_id): 
        worker_seed = torch.initial_seed() % 2**32 # กำหนด seed
        np.random.seed(worker_seed) # กำหนด seed สำหรับ NumPy
        random.seed(worker_seed) # กำหนด seed สำหรับ random

    g = torch.Generator() # สร้าง generator สำหรับการสุ่ม
    g.manual_seed(0) # กำหนด seed
    
    # Load video paths and labels  # โหลดที่อยู่วิดีโอและป้ายกำกับ
    video_paths, labels = load_videos_from_folders(folder_paths)

    # Split the data into train, validation, and test sets
    # แบ่งข้อมูลเป็นชุดฝึก, validation และทดสอบ
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        video_paths, labels, val_ratio=0.1, test_ratio=0.2
    )
    # Dictionary to store results
    # สร้าง dictionary สำหรับเก็บผลลัพธ์
    batch_size_results = {}

    # Device configuration
    # Device configuration # ตรวจสอบอุปกรณ์
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Experiment with different batch sizes
    # ทดลองขนาด batch ที่แตกต่างกัน
    for batch_size in batch_sizes:
        print(f"\n--- Experimenting with Batch Size: {batch_size} ---")
        
        # Store multiple run results for this batch size
        # เก็บผลลัพธ์การทดลองหลายรอบสำหรับขนาด batch นี้
        run_results = []

        for run in tqdm(range(num_runs), desc=f"Batch Size {batch_size} Runs"):
            print(f"\nRun {run + 1} of {num_runs}")
            
            # Create datasets
            # สร้าง datasets
            train_dataset = HandGestureDataset(train_paths, train_labels, transform=transform)
            val_dataset = HandGestureDataset(val_paths, val_labels, transform=transform)
            test_dataset = HandGestureDataset(test_paths, test_labels, transform=transform)

            # Create data loaders
            # สร้าง data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            # Initialize model
            # สร้างโมเดล
            model = CNN_LSTM().to(device)

            # Train model with validation set
            # ฝึกโมเดลโดยใช้ชุด validation
            model, best_acc = train_model(model, train_loader, val_loader, test_loader, num_epochs=10)
            
            # Store results for this run
            # เก็บผลลัพธ์สำหรับการทดลองนี้
            run_results.append(best_acc)

        # Calculate average accuracy for this batch size
        # คำนวณความแม่นยำเฉลี่ยสำหรับขนาด batch นี้
        avg_acc = sum(run_results) / num_runs
        batch_size_results[batch_size] = {
            'best_accuracies': run_results,
            'average_accuracy': avg_acc
        }

    # Print results
    print("\n--- Batch Size Experiment Results ---")
    for bs, results in batch_size_results.items():
        print(f"Batch Size {bs}:")
        print(f"  Best Accuracies: {[f'{acc:.2f}%' for acc in results['best_accuracies']]}")
        print(f"  Average Accuracy: {results['average_accuracy']:.2f}%")

    # Find the best batch size
    best_batch_size = max(batch_size_results, key=lambda k: batch_size_results[k]['average_accuracy'])
    print(f"\nBest Batch Size: {best_batch_size} (Average Accuracy: {batch_size_results[best_batch_size]['average_accuracy']:.2f}%)")

    return batch_size_results #  # คืนค่าผลลลัพธ์ของขนาด batch

def main():
    # Run batch size experiment
    experiment_batch_sizes() # ทดสอบการทดลองขนาด batch 

if __name__ == "__main__": # ถ้ารันไฟล์นี้เป็นโปรแกรมหลัก
    main() #เรียกใช้ฟังก์ชันหลัก
