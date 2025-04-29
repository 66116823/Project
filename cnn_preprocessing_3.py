#
import cv2  # นำเข้า OpenCV สำหรับการจัดการวิดีโอ
import numpy as np  # นำเข้า NumPy สำหรับการคำนวณเชิงตัวเลข
from pathlib import Path  # นำเข้า Path สำหรับการจัดการไฟล์และโฟลเดอร์
from typing import Dict, List, Tuple, Optional  # นำเข้า typing สำหรับการกำหนดประเภทข้อมูล
from collections import defaultdict  # นำเข้า defaultdict สำหรับเก็บข้อมูล
from concurrent.futures import ThreadPoolExecutor  # นำเข้า ThreadPoolExecutor สำหรับการประมวลผลแบบขนาน
from tqdm import tqdm  # นำเข้า tqdm สำหรับการแสดง progress bar
from sklearn.model_selection import train_test_split  # นำเข้า train_test_split สำหรับการแบ่งชุดข้อมูล
import re  # นำเข้า re สำหรับการจัดการกับสตริง


class VideoPreprocessor:
    def __init__(self,
                 sequence_length: int = 15,
                 turnoff_ratio: float = 0.05,
                 max_workers: int = 8):
        self.sequence_length = sequence_length  # ความยาวของ sequence
        self.turnoff_ratio = turnoff_ratio  # อัตราในการตัดข้อมูลที่ไม่จำเป็น
        self.max_workers = max_workers  # จำนวน worker สำหรับการประมวลผลแบบขนาน

        # Define base cases and paths
        self.base_cases = ['No_Request', 'Person_1', 'Person_2', 'Person_1_and_Person_2']

        # self.regular_paths = {
        #     'No_Request': r"D:\cnn_lstm\data\No_Requests_resized_splitted",
        #     'Person_1': r"D:\cnn_lstm\data\Person_1_Requests_resized_splitted",
        #     'Person_2': r"D:\cnn_lstm\data\Person_2_Requests_resized_splitted",
        #     'Person_1_and_Person_2': r"D:\cnn_lstm\data\Person_1_and_Person_2_Requests_resized_splitted"
        # }
        #
        # self.turnoff_paths = {
        #     'No_Request': r"D:\HumanDetection\GoogleMediapipe\Turn_Off\D_No_Requests\D_No_Requests_resized_splitted",
        #     'Person_1': r"D:\HumanDetection\GoogleMediapipe\Turn_Off\D_Person_1_Requests_resized_splitted",
        #     'Person_2': r"D:\cnn_lstm\data\D_Person_2_Requests_resized_splitted",
        #     'Person_1_and_Person_2': r"D:\cnn_lstm\data\D_Person_1_and_Person_2_Requests_resized_splitted"
        # }
                     
        # กำหนดเส้นทางของวิดีโอ
        self.regular_paths = {
            'No_Request': r"D:\cnn_lstm\No_Requests\No_Requests_resized\No_Requests_resized_splitted",
            'Person_1': r"D:\cnn_lstm\Person_1_Requests\Person_1_Requests_resized\Person_1_Requests_resized_splitted",
            'Person_2': r"D:\cnn_lstm\Person_2_Requests\Person_2_Requests_resized\Person_2_Requests_resized_splitted",
            'Person_1_and_Person_2': r"D:\cnn_lstm\Person_1_and_Person_2_Requests\Person_1_and_Person_2_Requests_resized\Person_1_and_Person_2_Requests_resized_splitted"
        }

        self.turnoff_paths = {

            'No_Request': r"D:\cnn_lstm\Turn_Off\D_No_Requests\D_No_Requests_resized\D_No_Requests_resized_splitted",
            'Person_1': r"D:\cnn_lstm\Turn_Off\D_Person_1_Requests\D_Person_1_Requests_resized\D_Person_1_Requests_resized_splitted",
            'Person_2': r"D:\cnn_lstm\Turn_Off\D_Person_2_Requests\D_Person_2_Requests_resized\D_Person_2_Requests_resized_splitted",
            'Person_1_and_Person_2': r"D:\cnn_lstm\Turn_Off\D_Person_1_and_Person_2_Requests\D_Person_1_and_Person_2_Requests_resized\D_Person_1_and_Person_2_Requests_resized_splitted"
        }

        # กำหนด mapping ระหว่าง case และ label
        self.case_to_label = {
            case: idx for idx, case in enumerate(self.base_cases)
        }

        # กำหนด mapping สำหรับเพศ
        self.gender_map = {
            1: 'Male-Female',
            2: 'Female-Male',
            3: 'Male-Male',
            4: 'Female-Female',
            5: 'Male (single)',
            6: 'Female (single)'
        }

    def get_case_info(self, filename: str) -> Optional[Tuple[str, int]]:
        """Extract hand type and gender from video filename."""
        try:
            # ลบส่วนขยายและแยกชื่อ
            clean_name = re.sub(r'\.avi$', '', filename)  # ลบ .avi
            parts = clean_name.split('_')  # แยกตาม '_'

            if len(parts) < 2:
                return None  # ถ้าข้อมูลไม่เพียงพอ

            # ดึงประเภทมือ
            if 'No_Requests' in filename:
                hand_type = 'None'
            else:
                if parts[0] not in ['Left', 'Right', 'Both']:
                    return None  # ถ้าประเภทมือไม่ถูกต้อง
                hand_type = parts[0]

            # ดึงรหัสเพศจากส่วนสุดท้าย
            try:
                gender_code = int(parts[-1])  # แปลงเป็น int
                if gender_code not in self.gender_map:
                    return None  # ถ้ารหัสเพศไม่ถูกต้อง
            except (IndexError, ValueError):
                return None  # ถ้าข้อมูลไม่ถูกต้อง

            return hand_type, gender_code  # คืนค่าประเภทมือและรหัสเพศ

        except Exception:
            return None  # คืนค่า None ถ้ามีข้อผิดพลาด

    def load_video(self, video_path: str) -> np.ndarray:
        """Load video frames from file."""
        try:
            cap = cv2.VideoCapture(str(video_path))  # เปิดวิดีโอ
            if not cap.isOpened():
                return np.array([])  # คืนค่า array ว่างถ้าไม่สามารถเปิดวิดีโอได้

            frames = []  # รายการสำหรับเก็บเฟรม
            while True:
                ret, frame = cap.read()  # อ่านเฟรม
                if not ret:
                    break  # ถ้าไม่สามารถอ่านเฟรมได้ ให้หยุด
                frames.append(frame)  # เก็บเฟรม

            cap.release()  # ปิดวิดีโอ

            if not frames:
                return np.array([])  # คืนค่า array ว่างถ้าไม่มีเฟรม

            return np.array(frames)  # คืนค่า array ของเฟรม

        except Exception as e:
            print(f"Error loading video {video_path}: {e}")  # แสดงข้อผิดพลาด
            return np.array([])  # คืนค่า array ว่างถ้ามีข้อผิดพลาด

    def extract_sequences(self, frames: np.ndarray) -> List[np.ndarray]:
        """Extract fixed-length sequences from video frames."""
        sequences = []  # รายการสำหรับเก็บ sequences
        total_frames = len(frames)  # จำนวนเฟรมทั้งหมด

        # คำนวณ sequences ที่สมบูรณ์
        for i in range(0, total_frames - self.sequence_length + 1, self.sequence_length):
            sequence = frames[i:i + self.sequence_length]  # ดึง sequence
            if len(sequence) == self.sequence_length:
                sequences.append(sequence)  # เพิ่ม sequence

        # จัดการกับเฟรมที่เหลือถ้ามี
        remaining_frames = total_frames % self.sequence_length
        if remaining_frames > 0:
            last_sequence = frames[-self.sequence_length:]  # ดึง sequence สุดท้าย
            if len(last_sequence) == self.sequence_length:
                sequences.append(last_sequence)  # เพิ่ม sequence สุดท้าย

        return sequences  # คืนค่ารายการ sequences

    def load_and_process_videos(self, folder_path: str) -> Dict:
        """Load and process videos from a folder with parallel processing."""
        video_data = defaultdict(lambda: defaultdict(list))  # สร้าง dict สำหรับเก็บข้อมูลวิดีโอ

        if not Path(folder_path).exists():  # เช็คว่า path มีอยู่หรือไม่
            print(f"❌ Path does not exist: {folder_path}")
            return video_data  # คืนค่า dict ว่างถ้า path ไม่ถูกต้อง

        def process_video(video_path: Path):
            case_info = self.get_case_info(video_path.name)  # ดึงข้อมูลประเภทมือและเพศ
            if case_info is None:
                return None  # คืนค่า None ถ้าข้อมูลไม่ถูกต้อง

            hand_type, gender_code = case_info  # ดึงประเภทมือและรหัสเพศ
            frames = self.load_video(str(video_path))  # โหลดเฟรมวิดีโอ

            if len(frames) == 0:
                return None  # คืนค่า None ถ้าไม่มีเฟรม

            sequences = self.extract_sequences(frames)  # ดึง sequences
            return hand_type, gender_code, sequences  # คืนค่าข้อมูล

        video_paths = list(Path(folder_path).glob('*.avi'))  # ดึง path ของวิดีโอในโฟลเดอร์

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:  # ใช้ ThreadPoolExecutor
            results = list(tqdm(  # แสดง progress bar
                executor.map(process_video, video_paths),  # ประมวลผลวิดีโอแบบขนาน
                total=len(video_paths),
                desc=f"Processing videos in {Path(folder_path).name}"
            ))

        for result in results:
            if result is not None:
                hand_type, gender_code, sequences = result  # ดึงข้อมูลจากผลลัพธ์
                video_data[hand_type][gender_code].extend(sequences)  # เพิ่ม sequences ใน dict

        return video_data  # คืนค่าข้อมูลวิดีโอ

    def load_and_balance_data(self):
        """Load and balance data from all folders."""
        print("\n📊 Loading and processing videos:")

        # ประมวลผลข้อมูลปกติและข้อมูล turnoff
        combined_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # สร้าง dict สำหรับเก็บข้อมูลรวม

        for main_case in self.base_cases:
            # ประมวลผลวิดีโอปกติ
            regular_data = self.load_and_process_videos(self.regular_paths[main_case])
            for hand_type, genders in regular_data.items():
                for gender, sequences in genders.items():
                    combined_data[main_case][hand_type][gender].extend(sequences)  # เพิ่ม sequences ใน dict

            # ประมวลผลวิดีโอ turnoff
            turnoff_data = self.load_and_process_videos(self.turnoff_paths[main_case])
            for hand_type, genders in turnoff_data.items():
                for gender, sequences in genders.items():
                    n_samples = max(1, int(len(sequences) * self.turnoff_ratio))  # คำนวณจำนวนตัวอย่างที่ต้องการ
                    sampled_indices = np.random.choice(len(sequences), n_samples, replace=False)  # เลือกตัวอย่างแบบสุ่ม
                    sampled_sequences = [sequences[i] for i in sampled_indices]  # ดึง sequences ที่เลือก
                    combined_data[main_case][hand_type][gender].extend(sampled_sequences)  # เพิ่ม sequences ใน dict

        return self._balance_data(combined_data)  # คืนค่าข้อมูลที่สมดุล
        
    def _balance_data(self, data_dict):
        """Balance data across all cases, hand types, and genders."""
        print("\n🔄 Balancing combined data:")

        # คำนวณสถิติสำหรับแต่ละกรณี
        case_stats = {}
        for main_case in self.base_cases:
            normal_total = 0  # จำนวน sequences ปกติ
            abnormal_total = 0  # จำนวน sequences ที่ผิดปกติ

            for hand_type, genders in data_dict[main_case].items():
                for gender, sequences in genders.items():
                    n_sequences = len(sequences)  # จำนวน sequences
                    n_abnormal = int(n_sequences * self.turnoff_ratio)  # คำนวณจำนวน sequences ที่ผิดปกติ
                    abnormal_total += n_abnormal  # อัปเดตจำนวนที่ผิดปกติ
                    normal_total += (n_sequences - n_abnormal)  # อัปเดตจำนวนที่ปกติ

            case_stats[main_case] = {
                'normal': normal_total,
                'abnormal': abnormal_total,
                'total': normal_total + abnormal_total
            }

            print(f"\nTotal sequences in {main_case}:")
            print(f"  Normal: {normal_total}")
            print(f"  Abnormal: {abnormal_total}")
            print(f"  Total: {normal_total + abnormal_total}")

        # สมดุลข้อมูล
        balanced_sequences = []  # รายการสำหรับเก็บ sequences ที่สมดุล
        balanced_labels = []  # รายการสำหรับเก็บ labels ที่สมดุล

        valid_cases = {k: v for k, v in case_stats.items() if v['total'] > 0}  # กรองกรณีที่มีข้อมูล
        if not valid_cases:
            raise ValueError("No valid sequences found in any case")  # แจ้งข้อผิดพลาดถ้าไม่มีข้อมูล

        min_normal = min(stats['normal'] for stats in valid_cases.values())  # คำนวณจำนวนที่น้อยที่สุดสำหรับ sequences ปกติ
        min_abnormal = min(stats['abnormal'] for stats in valid_cases.values())  # คำนวณจำนวนที่น้อยที่สุดสำหรับ sequences ที่ผิดปกติ
        
        for main_case in self.base_cases:
            if main_case not in valid_cases:  # ถ้ากรณีไม่ถูกต้อง ให้ข้าม
                continue

            print(f"\nBalancing {main_case}:")  # แสดงกรณีที่กำลังสมดุล
            hand_types = data_dict[main_case]  # ดึงประเภทมือ

            valid_hand_types = {
                ht: g for ht, g in hand_types.items()
                if any(len(s) > 0 for s in g.values())  # กรองประเภทมือที่มี sequences
            }

            sequences_per_hand = {  # คำนวณจำนวน sequences ที่ต้องการสำหรับแต่ละประเภทมือ
                'normal': min_normal // len(valid_hand_types),
                'abnormal': min_abnormal // len(valid_hand_types)
            }

            for hand_type, genders in valid_hand_types.items():  # วนรอบผ่านประเภทมือ
                valid_genders = {g: s for g, s in genders.items() if len(s) > 0}  # กรองเพศที่มี sequences
                sequences_per_gender = {  # คำนวณจำนวน sequences ที่ต้องการสำหรับแต่ละเพศ
                    'normal': sequences_per_hand['normal'] // len(valid_genders),
                    'abnormal': sequences_per_hand['abnormal'] // len(valid_genders)
                }

                for gender, sequences in valid_genders.items():  # วนรอบผ่านเพศ
                    if not sequences:  # ถ้าไม่มี sequences ให้ข้าม
                        continue

                    total_sequences = len(sequences)  # จำนวน sequences ทั้งหมด
                    n_abnormal = int(total_sequences * self.turnoff_ratio)  # คำนวณจำนวน sequences ที่ผิดปกติ

                    abnormal_seqs = sequences[-n_abnormal:]  # ดึง sequences ที่ผิดปกติ
                    normal_seqs = sequences[:-n_abnormal]  # ดึง sequences ที่ปกติ

                    # ตัวอย่าง sequences แบบสุ่มถ้าจำเป็น
                    selected_normal = self._sample_sequences(
                        normal_seqs,
                        sequences_per_gender['normal']
                    )
                    selected_abnormal = self._sample_sequences(
                        abnormal_seqs,
                        sequences_per_gender['abnormal']
                    )

                    selected_sequences = selected_normal + selected_abnormal  # รวม sequences ที่เลือก
                    balanced_sequences.extend(selected_sequences)  # เพิ่มในรายการที่สมดุล
                    balanced_labels.extend([self.case_to_label[main_case]] * len(selected_sequences))  # เพิ่ม labels

                    print(f"    {self.gender_map[gender]}: {len(selected_sequences)} sequences "
                          f"({len(selected_normal)} normal, {len(selected_abnormal)} abnormal)")

        return np.array(balanced_sequences), np.array(balanced_labels)  # คืนค่าข้อมูลที่สมดุล

    def _sample_sequences(self, sequences: List[np.ndarray], n_samples: int) -> List[np.ndarray]:
        """Sample sequences with replacement if needed."""
        if len(sequences) >= n_samples:
            indices = np.random.choice(len(sequences), n_samples, replace=False)  # เลือกแบบไม่ซ้ำ
        else:
            indices = np.random.choice(len(sequences), n_samples, replace=True)  # เลือกแบบซ้ำถ้าจำเป็น
        return [sequences[i] for i in indices]  # คืนค่ารายการ sequences ที่เลือก
    #
    def get_train_test_data(self):
        """Load, balance and split data into train, validation, and test sets."""
        X, y = self.load_and_balance_data()  # โหลดและสมดุลข้อมูล

        # แบ่งข้อมูลเป็น train+val และ test (80/20)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y  # แบ่งตามการกระจายของ labels
        )

        # แบ่ง train+val เป็น train และ val (80/20 ของที่เหลือ)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.1,
            random_state=42,
            stratify=y_temp  # แบ่งตามการกระจายของ labels
        )

        self.print_split_summary(X_train, X_val, X_test, y_train, y_val, y_test)  # แสดงสรุปการแบ่งข้อมูล
        return X_train, X_val, X_test, y_train, y_val, y_test  # คืนค่าข้อมูลที่แบ่งแล้ว


    def print_split_summary(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Print summary of data splits with class distribution."""
        total_samples = len(X_train) + len(X_val) + len(X_test)  # คำนวณจำนวนตัวอย่างทั้งหมด

        print("\nData Split Summary:")  # แสดงหัวข้อสรุปการแบ่งข้อมูล
        print(f"Total samples: {total_samples}")  # แสดงจำนวนตัวอย่างทั้งหมด
        print(f"Training set: {len(X_train)} samples ({len(X_train) / total_samples * 100:.1f}%)")  # แสดงข้อมูลชุดฝึก
        print(f"Validation set: {len(X_val)} samples ({len(X_val) / total_samples * 100:.1f}%)")  # แสดงข้อมูลชุดตรวจสอบ
        print(f"Test set: {len(X_test)} samples ({len(X_test) / total_samples * 100:.1f}%)")  # แสดงข้อมูลชุดทดสอบ

        def print_class_distribution(y, set_name):
            unique_labels, counts = np.unique(y, return_counts=True)
            print(f"\n{set_name} set:") # แสดงชื่อชุดข้อมูล
            for label, count in zip(unique_labels, counts): # วนรอบผ่านแต่ละคลาส
                class_name = next(name for name, idx in self.case_to_label.items() if idx == label) # หาชื่อคลาส
                percentage = count / len(y) * 100 # คำนวณเปอร์เซ็นต์ของคลาส
                print(f"Class {class_name}: {count} samples ({percentage:.1f}%)") # แสดงจำนวนและเปอร์เซ็นต์ของคลาส
 
        print("\nClass distribution:")  # แสดงหัวข้อการกระจายของคลาส
        print_class_distribution(y_train, "Training")  # แสดงการกระจายของคลาสในชุดฝึก
        print_class_distribution(y_val, "Validation")  # แสดงการกระจายของคลาสในชุดตรวจสอบ
        print_class_distribution(y_test, "Test")  # แสดงการกระจายของคลาสในชุดทดสอบ
