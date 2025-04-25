import os
import cv2

def reduce_frame_rate_and_resize(video_path, target_fps, target_width, target_height):
    # อ่านไฟล์วิดีโอ
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # ดึงข้อมูลเดิมจากวิดีโอ
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # คำนวณจำนวนเฟรมที่จะใช้
    frame_interval = int(original_fps / target_fps)
    if frame_interval <= 0:
        print("Target FPS is higher than or equal to the original FPS. No changes applied.")
        return

    # สร้างโฟลเดอร์ใหม่โดยใช้ชื่อโฟลเดอร์ล่าสุดของวิดีโอ
    base_folder = os.path.dirname(video_path)
    folder_name = os.path.basename(os.path.dirname(video_path))  # ชื่อโฟลเดอร์ล่าสุด
    new_folder = os.path.join(base_folder, f"{folder_name}_resized")
    os.makedirs(new_folder, exist_ok=True)
    
    # สร้างไฟล์ใหม่
    new_file_path = os.path.join(new_folder, os.path.basename(video_path))
    out = cv2.VideoWriter(new_file_path, fourcc, target_fps, (target_width, target_height))

    # print(f"Processing: {os.path.basename(video_path)}")  # แสดงแค่ชื่อไฟล์

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # เลือกเฉพาะเฟรมที่ตรงตามเงื่อนไข
        if frame_index % frame_interval == 0:
            # Resize เฟรม
            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)
        
        frame_index += 1

    # ปิดไฟล์และทำความสะอาด
    cap.release()
    out.release()
    print(f"Completed: {os.path.basename(video_path)}")  # แสดงแค่ชื่อไฟล์


def process_videos_in_folders(base_path, folders, target_fps, target_width, target_height):
    for folder in folders:
        full_path = os.path.join(base_path, folder)
        print(f"Processing folder: {full_path}")
        
        # เดินผ่านไฟล์ทั้งหมดในโฟลเดอร์
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.endswith((".avi", ".mp4")):  # กรองเฉพาะไฟล์วิดีโอ
                    video_path = os.path.join(root, file)
                    reduce_frame_rate_and_resize(video_path, target_fps, target_width, target_height)

# กำหนด path และโฟลเดอร์ที่ต้องการ
base_path = 'D:\\HumanDetection\\GoogleMediapipe\\'
folders = [
    'No_Requests',
    'Person_1_Requests',
    'Person_2_Requests',
    'Person_1_and_Person_2_Requests',
    'Turn_Off\\D_No_Requests',
    'Turn_Off\\D_Person_1_and_Person_2_Requests',
    'Turn_Off\\D_Person_1_Requests',
    'Turn_Off\\D_Person_2_Requests'
]

# เรียกฟังก์ชัน
process_videos_in_folders(base_path, folders, target_fps=1, target_width=256, target_height=192)
