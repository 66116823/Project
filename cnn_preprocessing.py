import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re


class VideoPreprocessor:
    def __init__(self,
                 sequence_length: int = 15,
                 turnoff_ratio: float = 0.1,
                 max_workers: int = 8):
        self.sequence_length = sequence_length
        self.turnoff_ratio = turnoff_ratio
        self.max_workers = max_workers

        # Define base cases and paths
        self.base_cases = ['No_Request', 'Person_1', 'Person_2', 'Person_1_and_Person_2']


        self.regular_paths = {
            'No_Request': r"D:\cnn_lstm\No_Requests",
            'Person_1': r"D:\cnn_lstm\Person_1_Requests",
            'Person_2': r"D:\cnn_lstm\Person_2_Requests",
            'Person_1_and_Person_2': r"D:\cnn_lstm\Person_1_and_Person_2_Requests"
        }
        self.turnoff_paths = {

            'No_Request': r"D:\cnn_lstm\Turnoff\D_No_Requests",
            'Person_1': r"D:\cnn_lstm\Turnoff\D_Person_1_Requests",
            'Person_2': r"D:\cnn_lstm\Turnoff\D_Person_2_Requests",
            'Person_1_and_Person_2': r"D:\cnn_lstm\Turnoff\D_Person_1_and_Person_2_Requests"
        }

        self.case_to_label = {
            case: idx for idx, case in enumerate(self.base_cases)
        }

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
            # Remove extension and split
            clean_name = re.sub(r'\.avi$', '', filename)
            parts = clean_name.split('_')

            if len(parts) < 2:
                return None

            # Extract hand type
            if 'No_Requests' in filename:
                hand_type = 'None'
            else:
                if parts[0] not in ['Left', 'Right', 'Both']:
                    return None
                hand_type = parts[0]

            # Extract gender code from last part
            try:
                gender_code = int(parts[-1])
                if gender_code not in self.gender_map:
                    return None
            except (IndexError, ValueError):
                return None

            return hand_type, gender_code

        except Exception:
            return None

    def load_video(self, video_path: str) -> np.ndarray:
        """Load video frames from file."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return np.array([])

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            if not frames:
                return np.array([])

            return np.array(frames)

        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return np.array([])

    def extract_sequences(self, frames: np.ndarray) -> List[np.ndarray]:
        """Extract fixed-length sequences from video frames."""
        sequences = []
        total_frames = len(frames)

        # Calculate complete sequences
        for i in range(0, total_frames - self.sequence_length + 1, self.sequence_length):
            sequence = frames[i:i + self.sequence_length]
            if len(sequence) == self.sequence_length:
                sequences.append(sequence)

        # Handle remaining frames if any
        remaining_frames = total_frames % self.sequence_length
        if remaining_frames > 0:
            last_sequence = frames[-self.sequence_length:]
            if len(last_sequence) == self.sequence_length:
                sequences.append(last_sequence)

        return sequences

    def load_and_process_videos(self, folder_path: str) -> Dict:
        """Load and process videos from a folder with parallel processing."""
        video_data = defaultdict(lambda: defaultdict(list))

        if not Path(folder_path).exists():
            print(f"❌ Path does not exist: {folder_path}")
            return video_data

        def process_video(video_path: Path):
            case_info = self.get_case_info(video_path.name)
            if case_info is None:
                return None

            hand_type, gender_code = case_info
            frames = self.load_video(str(video_path))

            if len(frames) == 0:
                return None

            sequences = self.extract_sequences(frames)
            return hand_type, gender_code, sequences

        video_paths = list(Path(folder_path).glob('*.avi'))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(process_video, video_paths),
                total=len(video_paths),
                desc=f"Processing videos in {Path(folder_path).name}"
            ))

        for result in results:
            if result is not None:
                hand_type, gender_code, sequences = result
                video_data[hand_type][gender_code].extend(sequences)

        return video_data

    def load_and_balance_data(self):
        """Load and balance data from all folders."""
        print("\n📊 Loading and processing videos:")

        # Process regular and turnoff data
        combined_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for main_case in self.base_cases:
            # Process regular videos
            regular_data = self.load_and_process_videos(self.regular_paths[main_case])
            for hand_type, genders in regular_data.items():
                for gender, sequences in genders.items():
                    combined_data[main_case][hand_type][gender].extend(sequences)

            # Process turnoff videos
            turnoff_data = self.load_and_process_videos(self.turnoff_paths[main_case])
            for hand_type, genders in turnoff_data.items():
                for gender, sequences in genders.items():
                    n_samples = max(1, int(len(sequences) * self.turnoff_ratio))
                    sampled_indices = np.random.choice(len(sequences), n_samples, replace=False)
                    sampled_sequences = [sequences[i] for i in sampled_indices]
                    combined_data[main_case][hand_type][gender].extend(sampled_sequences)

        return self._balance_data(combined_data)

    def _balance_data(self, data_dict):
        """Balance data across all cases, hand types, and genders."""
        print("\n🔄 Balancing combined data:")

        # Calculate statistics for each case
        case_stats = {}
        for main_case in self.base_cases:
            normal_total = 0
            abnormal_total = 0

            for hand_type, genders in data_dict[main_case].items():
                for gender, sequences in genders.items():
                    n_sequences = len(sequences)
                    n_abnormal = int(n_sequences * self.turnoff_ratio)
                    abnormal_total += n_abnormal
                    normal_total += (n_sequences - n_abnormal)

            case_stats[main_case] = {
                'normal': normal_total,
                'abnormal': abnormal_total,
                'total': normal_total + abnormal_total
            }

            print(f"\nTotal sequences in {main_case}:")
            print(f"  Normal: {normal_total}")
            print(f"  Abnormal: {abnormal_total}")
            print(f"  Total: {normal_total + abnormal_total}")

        # Balance data
        balanced_sequences = []
        balanced_labels = []

        valid_cases = {k: v for k, v in case_stats.items() if v['total'] > 0}
        if not valid_cases:
            raise ValueError("No valid sequences found in any case")

        min_normal = min(stats['normal'] for stats in valid_cases.values())
        min_abnormal = min(stats['abnormal'] for stats in valid_cases.values())

        for main_case in self.base_cases:
            if main_case not in valid_cases:
                continue

            print(f"\nBalancing {main_case}:")
            hand_types = data_dict[main_case]

            valid_hand_types = {
                ht: g for ht, g in hand_types.items()
                if any(len(s) > 0 for s in g.values())
            }

            sequences_per_hand = {
                'normal': min_normal // len(valid_hand_types),
                'abnormal': min_abnormal // len(valid_hand_types)
            }

            for hand_type, genders in valid_hand_types.items():
                valid_genders = {g: s for g, s in genders.items() if len(s) > 0}
                sequences_per_gender = {
                    'normal': sequences_per_hand['normal'] // len(valid_genders),
                    'abnormal': sequences_per_hand['abnormal'] // len(valid_genders)
                }

                for gender, sequences in valid_genders.items():
                    if not sequences:
                        continue

                    total_sequences = len(sequences)
                    n_abnormal = int(total_sequences * self.turnoff_ratio)

                    abnormal_seqs = sequences[-n_abnormal:]
                    normal_seqs = sequences[:-n_abnormal]

                    # Sample sequences with replacement if needed
                    selected_normal = self._sample_sequences(
                        normal_seqs,
                        sequences_per_gender['normal']
                    )
                    selected_abnormal = self._sample_sequences(
                        abnormal_seqs,
                        sequences_per_gender['abnormal']
                    )

                    selected_sequences = selected_normal + selected_abnormal
                    balanced_sequences.extend(selected_sequences)
                    balanced_labels.extend([self.case_to_label[main_case]] * len(selected_sequences))

                    print(f"    {self.gender_map[gender]}: {len(selected_sequences)} sequences "
                          f"({len(selected_normal)} normal, {len(selected_abnormal)} abnormal)")

        return np.array(balanced_sequences), np.array(balanced_labels)

    def _sample_sequences(self, sequences: List[np.ndarray], n_samples: int) -> List[np.ndarray]:
        """Sample sequences with replacement if needed."""
        if len(sequences) >= n_samples:
            indices = np.random.choice(len(sequences), n_samples, replace=False)
        else:
            indices = np.random.choice(len(sequences), n_samples, replace=True)
        return [sequences[i] for i in indices]

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
        X, y = self.load_and_balance_data()

        # Split into train+val and test (80/20)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Split train+val into train and val (80/20 of remaining)
        validation_size = 0.1 / 0.8  # คิดเป็น 12.5% ของ X_temp เพื่อให้เท่ากับ 10% ของข้อมูลทั้งหมด
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_size,
            random_state=42,
            stratify=y_temp
        )

        self.print_split_summary(X_train, X_val, X_test, y_train, y_val, y_test)

        # # Normalize ข้อมูล
        # X_train = self.normalize_sequences(X_train, fit=True)  # Compute & store statistics
        # X_val = self.normalize_sequences(X_val, fit=False)  # Use training statistics
        # X_test = self.normalize_sequences(X_test, fit=False)  # Use training statistics

        return X_train, X_val, X_test, y_train, y_val, y_test



    def print_split_summary(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Print summary of data splits with class distribution."""
        total_samples = len(X_train) + len(X_val) + len(X_test)

        print("\nData Split Summary:")
        print(f"Total samples: {total_samples}")
        print(f"Training set: {len(X_train)} samples ({len(X_train) / total_samples * 100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val) / total_samples * 100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({len(X_test) / total_samples * 100:.1f}%)")

        def print_class_distribution(y, set_name):
            unique_labels, counts = np.unique(y, return_counts=True)
            print(f"\n{set_name} set:")
            for label, count in zip(unique_labels, counts):
                class_name = next(name for name, idx in self.case_to_label.items() if idx == label)
                percentage = count / len(y) * 100
                print(f"Class {class_name}: {count} samples ({percentage:.1f}%)")

        print("\nClass distribution:")
        print_class_distribution(y_train, "Training")
        print_class_distribution(y_val, "Validation")
        print_class_distribution(y_test, "Test")

        # print("\nData statistics after normalization:")
        # print(f"Training set mean: {np.mean(X_train):.3f}")
        # print(f"Training set std: {np.std(X_train):.3f}")
        # print(f"Test set mean: {np.mean(X_test):.3f}")
        # print(f"Test set std: {np.std(X_test):.3f}")


