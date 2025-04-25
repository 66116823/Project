import cv2  # นำเข้าไลบรารี OpenCV สำหรับการประมวลผลภาพและวิดีโอ
import numpy as np  # นำเข้าไลบรารี NumPy สำหรับการจัดการอาร์เรย์
from pathlib import Path  # นำเข้า Path สำหรับการจัดการไฟล์และไดเรกทอรี
from typing import Dict, List, Tuple, Optional  # นำเข้าคลาสสำหรับการระบุประเภทข้อมูล
from collections import defaultdict  # นำเข้า defaultdict สำหรับการสร้างดิกชันนารีที่มีค่าเริ่มต้น
from concurrent.futures import ThreadPoolExecutor  # นำเข้า ThreadPoolExecutor สำหรับการทำงานแบบขนาน
from tqdm import tqdm  # นำเข้า tqdm สำหรับการแสดงแถบความก้าวหน้า
from sklearn.model_selection import train_test_split  # นำเข้า train_test_split สำหรับการแบ่งชุดข้อมูล
import re  # นำเข้าไลบรารี re สำหรับการจัดการสตริงที่มีรูปแบบ

class VideoPreprocessor:  # กำหนดคลาส VideoPreprocessor
    def __init__(self,  # ฟังก์ชันสร้างของคลาส
                 sequence_length: int = 15,  # ความยาวของลำดับ (จำนวนเฟรม)
                 turnoff_ratio: float = 0.1,  # อัตราส่วนของข้อมูลที่เป็น "Turnoff"
                 max_workers: int = 8):  # จำนวนงานสูงสุดที่สามารถทำงานพร้อมกัน
        self.sequence_length = sequence_length  # เก็บความยาวของลำดับ
        self.turnoff_ratio = turnoff_ratio  # เก็บอัตราส่วน Turnoff
        self.max_workers = max_workers  # เก็บจำนวนงานสูงสุดใน ThreadPool

        # กำหนดเคสพื้นฐานและที่อยู่
        self.base_cases = ['No_Request', 'Person_1', 'Person_2', 'Person_1_and_Person_2']  # ชื่อเคสต่างๆ

        # ที่อยู่ของข้อมูลปกติ
        self.regular_paths = {
            'No_Request': r"D:\cnn_lstm\No_Requests",
            'Person_1': r"D:\cnn_lstm\Person_1_Requests",
            'Person_2': r"D:\cnn_lstm\Person_2_Requests",
            'Person_1_and_Person_2': r"D:\cnn_lstm\Person_1_and_Person_2_Requests"
        }
        
        # ที่อยู่ของข้อมูล Turnoff
        self.turnoff_paths = {
            'No_Request': r"D:\cnn_lstm\Turnoff\D_No_Requests",
            'Person_1': r"D:\cnn_lstm\Turnoff\D_Person_1_Requests",
            'Person_2': r"D:\cnn_lstm\Turnoff\D_Person_2_Requests",
            'Person_1_and_Person_2': r"D:\cnn_lstm\Turnoff\D_Person_1_and_Person_2_Requests"
        }

        # แผนที่เคสไปยังเลเบล
        self.case_to_label = {
            case: idx for idx, case in enumerate(self.base_cases)  # สร้างดิกชันนารีที่เก็บเคสและดัชนี
        }

        # แผนที่รหัสเพศ
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
            # ลบส่วนขยายและแยกชื่อไฟล์
            clean_name = re.sub(r'\.avi$', '', filename)  # ลบ .avi ออกจากชื่อไฟล์
            parts = clean_name.split('_')  # แยกชื่อไฟล์ตาม '_'

            if len(parts) < 2:  # ตรวจสอบว่ามีส่วนที่เพียงพอหรือไม่
                return None  # คืนค่า None ถ้าไม่เพียงพอ

            # ดึงประเภทมือ
            if 'No_Requests' in filename:  # ถ้าชื่อไฟล์มี 'No_Requests'
                hand_type = 'None'  # ประเภทมือเป็น 'None'
            else:
                if parts[0] not in ['Left', 'Right', 'Both']:  # ตรวจสอบประเภทมือ
                    return None  # คืนค่า None ถ้าไม่ตรง
                hand_type = parts[0]  # กำหนดประเภทมือ

            # ดึงรหัสเพศจากส่วนสุดท้าย
            try:
                gender_code = int(parts[-1])  # แปลงส่วนสุดท้ายเป็นจำนวนเต็ม
                if gender_code not in self.gender_map:  # ตรวจสอบว่ารหัสเพศถูกต้องหรือไม่
                    return None  # คืนค่า None ถ้าไม่ถูกต้อง
            except (IndexError, ValueError):  # จัดการข้อผิดพลาด
                return None  # คืนค่า None

            return hand_type, gender_code  # คืนค่าประเภทมือและรหัสเพศ

        except Exception:  # จัดการข้อผิดพลาดทั่วไป
            return None  # คืนค่า None

    def load_video(self, video_path: str) -> np.ndarray:
        """Load video frames from file."""
        try:
            cap = cv2.VideoCapture(str(video_path))  # เปิดวิดีโอจากที่อยู่ที่ระบุ
            if not cap.isOpened():  # ตรวจสอบว่าการเปิดสำเร็จหรือไม่
                return np.array([])  # คืนค่าอาร์เรย์ว่างถ้าไม่สำเร็จ

            frames = []  # สร้างรายการสำหรับเก็บเฟรม
            while True:  # วนลูปเพื่ออ่านเฟรม
                ret, frame = cap.read()  # อ่านเฟรม
                if not ret:  # ถ้าไม่อ่านได้
                    break  # ออกจากลูป
                frames.append(frame)  # เพิ่มเฟรมในรายการ

            cap.release()  # ปิดการเชื่อมต่อกับวิดีโอ

            if not frames:  # ถ้าไม่มีเฟรม
                return np.array([])  # คืนค่าอาร์เรย์ว่าง

            return np.array(frames)  # คืนค่าเฟรมเป็นอาร์เรย์ NumPy

        except Exception as e:  # จัดการข้อผิดพลาด
            print(f"Error loading video {video_path}: {e}")  # แสดงข้อความผิดพลาด
            return np.array([])  # คืนค่าอาร์เรย์ว่าง

    def extract_sequences(self, frames: np.ndarray) -> List[np.ndarray]:
        """Extract fixed-length sequences from video frames."""
        sequences = []  # สร้างรายการสำหรับเก็บลำดับ
        total_frames = len(frames)  # จำนวนเฟรมทั้งหมด

        # คำนวณลำดับที่สมบูรณ์
        for i in range(0, total_frames - self.sequence_length + 1, self.sequence_length):
            sequence = frames[i:i + self.sequence_length]  # ดึงลำดับของเฟรม
            if len(sequence) == self.sequence_length:  # ตรวจสอบว่ามีความยาวตรงตามที่กำหนด
                sequences.append(sequence)  # เพิ่มลำดับในรายการ

        # จัดการเฟรมที่เหลือถ้ามี
        remaining_frames = total_frames % self.sequence_length  # คำนวณจำนวนเฟรมที่เหลือ
        if remaining_frames > 0:  # ถ้ามีเฟรมที่เหลือ
            last_sequence = frames[-self.sequence_length:]  # ดึงลำดับสุดท้าย
            if len(last_sequence) == self.sequence_length:  # ตรวจสอบความยาว
                sequences.append(last_sequence)  # เพิ่มลำดับสุดท้ายในรายการ

        return sequences  # คืนค่าลำดับทั้งหมด

    def load_and_process_videos(self, folder_path: str) -> Dict:
        """Load and process videos from a folder with parallel processing."""
        video_data = defaultdict(lambda: defaultdict(list))  # สร้างดิกชันนารีสำหรับเก็บข้อมูลวิดีโอ

        if not Path(folder_path).exists():  # ตรวจสอบว่าโฟลเดอร์มีอยู่
            print(f"❌ Path does not exist: {folder_path}")  # แสดงข้อความผิดพลาด
            return video_data  # คืนค่าดิกชันนารีว่าง

        def process_video(video_path: Path):  # ฟังก์ชันย่อยสำหรับประมวลผลวิดีโอ
            case_info = self.get_case_info(video_path.name)  # ดึงข้อมูลเคสจากชื่อไฟล์
            if case_info is None:  # ถ้าข้อมูลเคสไม่ถูกต้อง
                return None  # คืนค่า None

            hand_type, gender_code = case_info  # แยกประเภทมือและรหัสเพศ
            frames = self.load_video(str(video_path))  # โหลดเฟรมจากวิดีโอ

            if len(frames) == 0:  # ถ้าไม่มีเฟรม
                return None  # คืนค่า None

            sequences = self.extract_sequences(frames)  # ดึงลำดับเฟรม
            return hand_type, gender_code, sequences  # คืนค่าข้อมูลต่างๆ

        video_paths = list(Path(folder_path).glob('*.avi'))  # ดึงรายการไฟล์วิดีโอในโฟลเดอร์

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:  # สร้าง ThreadPoolExecutor
            results = list(tqdm(  # แสดงแถบความก้าวหน้า
                executor.map(process_video, video_paths),  # ประมวลผลวิดีโอแบบขนาน
                total=len(video_paths),  # จำนวนทั้งหมด
                desc=f"Processing videos in {Path(folder_path).name}"  # คำอธิบาย
            ))

        for result in results:  # วนลูปผ่านผลลัพธ์
            if result is not None:  # ถ้าผลลัพธ์ไม่ใช่ None
                hand_type, gender_code, sequences = result  # แยกข้อมูล
                video_data[hand_type][gender_code].extend(sequences)  # เพิ่มลำดับในดิกชันนารี

        return video_data  # คืนค่าข้อมูลวิดีโอทั้งหมด

    def load_and_balance_data(self):
        """Load and balance data from all folders."""
        print("\n📊 Loading and processing videos:")  # แสดงข้อความเริ่มโหลดข้อมูล

        # ประมวลผลข้อมูลปกติและ Turnoff
        combined_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # สร้างดิกชันนารีสำหรับข้อมูลรวม

        for main_case in self.base_cases:  # วนลูปผ่านเคสพื้นฐาน
            # ประมวลผลวิดีโอปกติ
            regular_data = self.load_and_process_videos(self.regular_paths[main_case])  # โหลดข้อมูลปกติ
            for hand_type, genders in regular_data.items():  # วนลูปผ่านประเภทมือและเพศ
                for gender, sequences in genders.items():  # วนลูปผ่านเพศ
                    combined_data[main_case][hand_type][gender].extend(sequences)  # เพิ่มลำดับในข้อมูลรวม

            # ประมวลผลวิดีโอ Turnoff
            turnoff_data = self.load_and_process_videos(self.turnoff_paths[main_case])  # โหลดข้อมูล Turnoff
            for hand_type, genders in turnoff_data.items():  # วนลูปผ่านประเภทมือและเพศ
                for gender, sequences in genders.items():  # วนลูปผ่านเพศ
                    n_samples = max(1, int(len(sequences) * self.turnoff_ratio))  # คำนวณจำนวนตัวอย่าง
                    sampled_indices = np.random.choice(len(sequences), n_samples, replace=False)  # เลือกดัชนีแบบสุ่ม
                    sampled_sequences = [sequences[i] for i in sampled_indices]  # ดึงลำดับที่เลือก
                    combined_data[main_case][hand_type][gender].extend(sampled_sequences)  # เพิ่มข้อมูล Turnoff

        return self._balance_data(combined_data)  # คืนค่าข้อมูลที่ถูกปรับสมดุล

    def _balance_data(self, data_dict):
        """Balance data across all cases, hand types, and genders."""
        print("\n🔄 Balancing combined data:")  # แสดงข้อความเริ่มปรับสมดุลข้อมูล

        # คำนวณสถิติสำหรับแต่ละเคส
        case_stats = {}  # สร้างดิกชันนารีสำหรับสถิติ
        for main_case in self.base_cases:  # วนลูปผ่านเคสพื้นฐาน
            normal_total = 0  # จำนวนลำดับปกติ
            abnormal_total = 0  # จำนวนลำดับผิดปกติ

            for hand_type, genders in data_dict[main_case].items():  # วนลูปผ่านประเภทมือและเพศ
                for gender, sequences in genders.items():  # วนลูปผ่านเพศ
                    n_sequences = len(sequences)  # จำนวนลำดับ
                    n_abnormal = int(n_sequences * self.turnoff_ratio)  # คำนวณจำนวนลำดับผิดปกติ
                    abnormal_total += n_abnormal  # เพิ่มจำนวนลำดับผิดปกติ
                    normal_total += (n_sequences - n_abnormal)  # เพิ่มจำนวนลำดับปกติ

            case_stats[main_case] = {  # เก็บสถิติในดิกชันนารี
                'normal': normal_total,
                'abnormal': abnormal_total,
                'total': normal_total + abnormal_total
            }

            print(f"\nTotal sequences in {main_case}:")  # แสดงข้อความสรุปจำนวนลำดับ
            print(f"  Normal: {normal_total}")  # แสดงจำนวนลำดับปกติ
            print(f"  Abnormal: {abnormal_total}")  # แสดงจำนวนลำดับผิดปกติ
            print(f"  Total: {normal_total + abnormal_total}")  # แสดงจำนวนลำดับทั้งหมด

        # ปรับสมดุลข้อมูล
        balanced_sequences = []  # สร้างรายการสำหรับลำดับที่สมดุล
        balanced_labels = []  # สร้างรายการสำหรับเลเบลที่สมดุล

        valid_cases = {k: v for k, v in case_stats.items() if v['total'] > 0}  # กรองเคสที่มีข้อมูล
        if not valid_cases:  # ถ้าไม่มีเคสที่ถูกต้อง
            raise ValueError("No valid sequences found in any case")  # ยกข้อผิดพลาด

        min_normal = min(stats['normal'] for stats in valid_cases.values())  # หาจำนวนลำดับปกติน้อยที่สุด
        min_abnormal = min(stats['abnormal'] for stats in valid_cases.values())  # หาจำนวนลำดับผิดปกติน้อยที่สุด

        for main_case in self.base_cases:  # วนลูปผ่านเคสพื้นฐาน
            if main_case not in valid_cases:  # ถ้าเคสไม่อยู่ใน valid_cases
                continue  # ข้ามไป

            print(f"\nBalancing {main_case}:")  # แสดงข้อความเริ่มปรับสมดุลเคส
            hand_types = data_dict[main_case]  # ดึงข้อมูลประเภทมือ

            valid_hand_types = {  # กรองประเภทมือที่มีข้อมูล
                ht: g for ht, g in hand_types.items()
                if any(len(s) > 0 for s in g.values())
            }

            sequences_per_hand = {  # กำหนดจำนวนลำดับต่อประเภทมือ
                'normal': min_normal // len(valid_hand_types),
                'abnormal': min_abnormal // len(valid_hand_types)
            }

            for hand_type, genders in valid_hand_types.items():  # วนลูปผ่านประเภทมือ
                valid_genders = {g: s for g, s in genders.items() if len(s) > 0}  # กรองเพศที่มีข้อมูล
                sequences_per_gender = {  # กำหนดจำนวนลำดับต่อเพศ
                    'normal': sequences_per_hand['normal'] // len(valid_genders),
                    'abnormal': sequences_per_hand['abnormal'] // len(valid_genders)
                }

                for gender, sequences in valid_genders.items():  # วนลูปผ่านเพศ
                    if not sequences:  # ถ้าไม่มีลำดับ
                        continue  # ข้ามไป

                    total_sequences = len(sequences)  # จำนวนลำดับทั้งหมด
                    n_abnormal = int(total_sequences * self.turnoff_ratio)  # จำนวนลำดับผิดปกติ

                    abnormal_seqs = sequences[-n_abnormal:]  # ดึงลำดับผิดปกติ
                    normal_seqs = sequences[:-n_abnormal]  # ดึงลำดับปกติ

                    # ตัวอย่างลำดับโดยอาจใช้การแทนที่ถ้าจำเป็น
                    selected_normal = self._sample_sequences(
                        normal_seqs,
                        sequences_per_gender['normal']
                    )  # ตัวอย่างลำดับปกติ
                    selected_abnormal = self._sample_sequences(
                        abnormal_seqs,
                        sequences_per_gender['abnormal']
                    )  # ตัวอย่างลำดับผิดปกติ

                    selected_sequences = selected_normal + selected_abnormal  # รวมลำดับที่เลือก
                    balanced_sequences.extend(selected_sequences)  # เพิ่มลำดับในรายการที่สมดุล
                    balanced_labels.extend([self.case_to_label[main_case]] * len(selected_sequences))  # เพิ่มเลเบลในรายการที่สมดุล

                    print(f"    {self.gender_map[gender]}: {len(selected_sequences)} sequences "
                          f"({len(selected_normal)} normal, {len(selected_abnormal)} abnormal)")  # แสดงจำนวนลำดับที่เลือก

        return np.array(balanced_sequences), np.array(balanced_labels)  # คืนค่าลำดับและเลเบลที่สมดุล

    def _sample_sequences(self, sequences: List[np.ndarray], n_samples: int) -> List[np.ndarray]:
        """Sample sequences with replacement if needed."""
        if len(sequences) >= n_samples:  # ถ้าจำนวนลำดับมากกว่าหรือเท่ากับจำนวนตัวอย่างที่ต้องการ
            indices = np.random.choice(len(sequences), n_samples, replace=False)  # เลือกดัชนีแบบสุ่มโดยไม่ให้ซ้ำ
        else:  # ถ้าจำนวนลำดับน้อยกว่าจำนวนตัวอย่างที่ต้องการ
            indices = np.random.choice(len(sequences), n_samples, replace=True)  # เลือกดัชนีแบบสุ่มโดยอนุญาตให้ซ้ำ
        return [sequences[i] for i in indices]  # คืนค่าลำดับที่เลือก

    # def normalize_sequences(self, sequences: np.ndarray, fit: bool = False) -> np.ndarray:
    #     """
    #     Normalize video sequences using min-max normalization to range [0,1].
    #
    #     Args:
    #         sequences: Video sequences of shape (n_sequences, sequence_length, height, width, channels)
    #         fit: Whether to compute new statistics (True for training) or use existing ones (False for val/test)
    #
    #     Returns:
    #         Normalized sequences with pixel values scaled to [0,1]
    #     """
    #     if fit:
    #         # Store statistics for later use with validation/test data
    #         self.pixel_min = sequences.min()
    #         self.pixel_max = sequences.max()
    #
    #     # Avoid division by zero
    #     denominator = (self.pixel_max - self.pixel_min)
    #     if denominator == 0:
    #         denominator = 1.0
    #
    #     # Apply min-max normalization
    #     normalized = (sequences - self.pixel_min) / denominator
    #
    #     return normalized

    def get_train_test_data(self):
        """Load, balance and split data into train, validation, and test sets."""
        X, y = self.load_and_balance_data()  # โหลดและปรับสมดุลข้อมูล

        # แบ่งข้อมูลเป็นชุดฝึก + ตรวจสอบและชุดทดสอบ (80/20)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.2,  # ขนาดของชุดทดสอบ
            random_state=42,  # รหัสสุ่มสำหรับการแบ่ง
            stratify=y  # แบ่งแบบ stratified เพื่อรักษาสัดส่วนของคลาส
        )

        # แบ่งชุดฝึก + ตรวจสอบเป็นชุดฝึกและชุดตรวจสอบ (80/20 ของที่เหลือ)
        validation_size = 0.1 / 0.8  # คำนวณขนาดชุดตรวจสอบให้เป็น 10% ของข้อมูลทั้งหมด
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_size,  # ขนาดของชุดตรวจสอบ
            random_state=42,  # รหัสสุ่ม
            stratify=y_temp  # แบ่งแบบ stratified
        )

        self.print_split_summary(X_train, X_val, X_test, y_train, y_val, y_test)  # แสดงสรุปการแบ่งข้อมูล

        # # Normalize ข้อมูล (ถูกคอมเมนต์ไว้)
        # X_train = self.normalize_sequences(X_train, fit=True)  # คำนวณและเก็บสถิติ
        # X_val = self.normalize_sequences(X_val, fit=False)  # ใช้สถิติจากการฝึก
        # X_test = self.normalize_sequences(X_test, fit=False)  # ใช้สถิติจากการฝึก

        return X_train, X_val, X_test, y_train, y_val, y_test  # คืนค่าชุดข้อมูลที่แบ่งแล้ว

    def print_split_summary(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Print summary of data splits with class distribution."""
        total_samples = len(X_train) + len(X_val) + len(X_test)  # คำนวณจำนวนตัวอย่างทั้งหมด

        print("\nData Split Summary:")  # แสดงหัวข้อสรุป
        print(f"Total samples: {total_samples}")  # แสดงจำนวนตัวอย่างทั้งหมด
        print(f"Training set: {len(X_train)} samples ({len(X_train) / total_samples * 100:.1f}%)")  # แสดงจำนวนชุดฝึก
        print(f"Validation set: {len(X_val)} samples ({len(X_val) / total_samples * 100:.1f}%)")  # แสดงจำนวนชุดตรวจสอบ
        print(f"Test set: {len(X_test)} samples ({len(X_test) / total_samples * 100:.1f}%)")  # แสดงจำนวนชุดทดสอบ

        def print_class_distribution(y, set_name):  # ฟังก์ชันย่อยแสดงการกระจายของคลาส
            unique_labels, counts = np.unique(y, return_counts=True)  # คำนวณจำนวนตัวอย่างแต่ละคลาส
            print(f"\n{set_name} set:")  # แสดงชื่อชุดข้อมูล
            for label, count in zip(unique_labels, counts):  # วนลูปผ่านคลาสและจำนวน
                class_name = next(name for name, idx in self.case_to_label.items() if idx == label)  # หาชื่อคลาส
                percentage = count / len(y) * 100  # คำนวณเปอร์เซ็นต์
                print(f"Class {class_name}: {count} samples ({percentage:.1f}%)")  # แสดงข้อมูลคลาส

        print("\nClass distribution:")  # แสดงหัวข้อการกระจายคลาส
        print_class_distribution(y_train, "Training")  # แสดงการกระจายชุดฝึก
        print_class_distribution(y_val, "Validation")  # แสดงการกระจายชุดตรวจสอบ
        print_class_distribution(y_test, "Test")  # แสดงการกระจายชุดทดสอบ

        # print("\nData statistics after normalization:")  # แสดงสถิติหลังการปรับมาตรฐาน (ถูกคอมเมนต์ไว้)
        # print(f"Training set mean: {np.mean(X_train):.3f}")  # แสดงค่าเฉลี่ยชุดฝึก
        # print(f"Training set std: {np.std(X_train):.3f}")  # แสดงค่ามาตรฐานชุดฝึก
        # print(f"Test set mean: {np.mean(X_test):.3f}")  # แสดงค่าเฉลี่ยชุดทดสอบ
        # print(f"Test set std: {np.std(X_test):.3f}")  # แสดงค่ามาตรฐานชุดทดสอบ
