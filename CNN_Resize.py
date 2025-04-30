import os  # นำเข้าโมดูล os สำหรับการจัดการไฟล์และไดเรกทอรี
import cv2  # นำเข้า OpenCV สำหรับการจัดการวิดีโอ

def reduce_frame_rate_and_resize(video_path, target_fps, target_width, target_height):
    # ฟังก์ชันเพื่อลดอัตราเฟรมและปรับขนาดวิดีโอ
    cap = cv2.VideoCapture(video_path) # อ่านไฟล์วิดีโอ
    if not cap.isOpened(): # ตรวจสอบว่าการเปิดไฟล์วิดีโอสำเร็จหรือไม่
        print(f"Error: Cannot open video file {video_path}") # แสดงข้อความผิดพลาดหากไม่สามารถเปิดได้
        return # ออกจากฟังก์ชัน
    
    # ดึงข้อมูลเดิมจากวิดีโอ
    original_fps = cap.get(cv2.CAP_PROP_FPS) # รับอัตราเฟรมเดิม
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # รับความกว้างของวิดีโอ
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # รับความสูงของวิดีโอ
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) # รับ codec ของวิดีโอ
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # รับจำนวนเฟรมทั้งหมด

    # คำนวณจำนวนเฟรมที่จะใช้
    frame_interval = int(original_fps / target_fps) # คำนวณช่วงเฟรมที่ต้องใช้
    if frame_interval <= 0: # หากอัตราเฟรมเป้าหมายสูงกว่าหรือเท่ากับอัตราเฟรมเดิม
        print("Target FPS is higher than or equal to the original FPS. No changes applied.") # แสดงข้อความเตือน
        return # ออกจากฟังก์ชัน

    # สร้างโฟลเดอร์ใหม่โดยใช้ชื่อโฟลเดอร์ล่าสุดของวิดีโอ
    base_folder = os.path.dirname(video_path) # รับเส้นทางไดเรกทอรีของวิดีโอ
    folder_name = os.path.basename(os.path.dirname(video_path))  # ชื่อโฟลเดอร์ล่าสุด
    new_folder = os.path.join(base_folder, f"{folder_name}_resized")  # กำหนดเส้นทางใหม่สำหรับโฟลเดอร์
    os.makedirs(new_folder, exist_ok=True) # สร้างโฟลเดอร์ใหม่หากยังไม่มี
     
    # สร้างไฟล์ใหม่
    new_file_path = os.path.join(new_folder, os.path.basename(video_path))  # กำหนดเส้นทางไฟล์ใหม่
    out = cv2.VideoWriter(new_file_path, fourcc, target_fps, (target_width, target_height)) # เริ่มต้นวิดีโอไรเตอร์สำหรับเอาต์พุต

    # print(f"Processing: {os.path.basename(video_path)}")  # แสดงแค่ชื่อไฟล์

    frame_index = 0 # กำหนดค่าดัชนีเฟรมเริ่มต้น
    while True: # วนลูปเพื่ออ่านเฟรม
        ret, frame = cap.read() # อ่านเฟรมจากวิดีโอ
        if not ret: # หากไม่สามารถอ่านเฟรมได้
            break # ออกจากลูป

        # เลือกเฉพาะเฟรมที่ตรงตามเงื่อนไข
        if frame_index % frame_interval == 0: # หากเฟรมตรงตามเงื่อนไข
            # Resize เฟรม 
            resized_frame = cv2.resize(frame, (target_width, target_height)) # ปรับขนาดเฟรม
            out.write(resized_frame)  # เขียนเฟรมที่ปรับขนาดไปยังไฟล์เอาต์พุต
        
        frame_index += 1 # เพิ่มดัชนีเฟรม

    # ปิดไฟล์และทำความสะอาด
    cap.release() # ปล่อยวิดีโอแคปเจอร์
    out.release() # ปล่อยวิดีโอไรเตอร์
    print(f"Completed: {os.path.basename(video_path)}")  # แสดงแค่ชื่อไฟล์ที่เสร็จสิ้น


def process_videos_in_folders(base_path, folders, target_fps, target_width, target_height): # ฟังก์ชันสำหรับประมวลผลวิดีโอในโฟลเดอร์
    for folder in folders:  # วนลูปผ่านแต่ละโฟลเดอร์
        full_path = os.path.join(base_path, folder) # รับเส้นทางเต็มของโฟลเดอร์
        print(f"Processing folder: {full_path}") # แสดงโฟลเดอร์ที่กำลังประมวลผล
        
        # เดินผ่านไฟล์ทั้งหมดในโฟลเดอร์
        for root, _, files in os.walk(full_path): # วนลูปผ่านไฟล์ในโฟลเดอร์
            for file in files: # วนลูปผ่านไฟล์แต่ละไฟล์
                if file.endswith((".avi", ".mp4")):  # กรองเฉพาะไฟล์วิดีโอ
                    video_path = os.path.join(root, file) # รับเส้นทางไฟล์ของวิดีโอ
                    reduce_frame_rate_and_resize(video_path, target_fps, target_width, target_height) # เรียกใช้ฟังก์ชันจัดการวิดีโอ

# กำหนด path และโฟลเดอร์ที่ต้องการ
base_path = 'D:\\HumanDetection\\GoogleMediapipe\\'  # กำหนดเส้นทางหลัก
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
process_videos_in_folders(base_path, folders, target_fps=1, target_width=256, target_height=192) # เรียกใช้ฟังก์ชันเพื่อประมวลผลวิดีโอ
