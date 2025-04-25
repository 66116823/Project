import random
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix

from cnn_preprocessing import load_video
from cnn_preprocessing import select_frames
from cnn_preprocessing import split_dataset
from cnn_preprocessing import folder_paths,load_videos_from_folders
from PIL import Image
from tqdm import tqdm


# Parameters
num_classes = 4


class HandGestureDataset(Dataset):
    def __init__(self, video_paths, labels, sequence_length=15, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]

            # Check if the video file exists
            if not Path(video_path).is_file():
                print(f"Warning: Video file not found - {video_path}")
                # Return a default or skip this sample
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

        return x


def train_model(model, train_loader, val_loader, test_loader, num_epochs=10, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Mixed precision training
    scaler = torch.amp.GradScaler()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_acc = 0.0
    no_improve = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)

            # Runs the forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Specify the device type
                outputs = model(sequences)
                loss = criterion(outputs, labels)

            # Scales loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()
            
            # Unscales gradients and calls optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            optimizer.zero_grad()

            # outputs = model(sequences)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Testing
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        test_acc = 100 * test_correct / test_total
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%')

        # Update best accuracy based on validation performance
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1

        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)

        # Early stopping
        if no_improve >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break

    return model, best_acc

def experiment_batch_sizes(batch_sizes=[8, 16, 32, 64], num_runs=3):
    # Use mixed precision training
    scaler = torch.amp.GradScaler()

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Optional: data augmentation for better generalization
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    # Use more efficient data loading
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    # Load video paths and labels
    video_paths, labels = load_videos_from_folders(folder_paths)

    # Split the data into train, validation, and test sets
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        video_paths, labels, val_ratio=0.1, test_ratio=0.2
    )
    # Dictionary to store results
    batch_size_results = {}

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Experiment with different batch sizes
    for batch_size in batch_sizes:
        print(f"\n--- Experimenting with Batch Size: {batch_size} ---")
        
        # Store multiple run results for this batch size
        run_results = []

        for run in tqdm(range(num_runs), desc=f"Batch Size {batch_size} Runs"):
            print(f"\nRun {run + 1} of {num_runs}")
            
            # Create datasets
            train_dataset = HandGestureDataset(train_paths, train_labels, transform=transform)
            val_dataset = HandGestureDataset(val_paths, val_labels, transform=transform)
            test_dataset = HandGestureDataset(test_paths, test_labels, transform=transform)

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            # Initialize model
            model = CNN_LSTM().to(device)

            # Train model with validation set
            model, best_acc = train_model(model, train_loader, val_loader, test_loader, num_epochs=10)
            
            # Store results for this run
            run_results.append(best_acc)

        # Calculate average accuracy for this batch size
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

    return batch_size_results

def main():
    # Run batch size experiment
    experiment_batch_sizes()

if __name__ == "__main__":
    main()
