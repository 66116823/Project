import os # นำเข้า os สำหรับการจัดการไฟล์และโฟลเดอร์
import cv2 # นำเข้า OpenCV สำหรับการประมวลผลภาพ

def split_video_into_clips(video_path, output_folder, clip_duration=15): # ฟังก์ชันสำหรับแบ่งวิดีโอเป็นคลิปตามระยะเวลาที่กำหนด
    cap = cv2.VideoCapture(video_path) # เปิดไฟล์วิดีโอ
    if not cap.isOpened(): # ตรวจสอบว่าการเปิดไฟล์วิดีโอสำเร็จหรือไม่
        print(f"Error: Cannot open video file {video_path}") # แสดงข้อความผิดพลาดหากไม่สามารถเปิดได้
        return # ออกจากฟังก์ชัน
    # รับข้อมูลเกี่ยวกับวิดีโอ
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # รับจำนวนเฟรมต่อวินาที
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # รับจำนวนเฟรมทั้งหมด
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # รับความกว้างของวิดีโอ
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # รับความสูงของวิดีโอ
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # กำหนด codec สำหรับวิดีโอเอาต์พุต
    frames_per_clip = clip_duration * fps  # คำนวณจำนวนเฟรมต่อคลิปตามระยะเวลา
    # คำนวณจำนวนคลิปที่คาดว่าจะได้
    expected_clips = (frame_count + frames_per_clip - 1) // frames_per_clip
    # แสดงข้อมูลการประมวลผลวิดีโอ
    print(f"\nProcessing video: {video_path}") # แสดงว่าวิดีโอกำลังถูกประมวลผล
    print(f"FPS: {fps}, Total Frames: {frame_count}, Resolution: {width}x{height}") # แสดงสถิติของวิดีโอ
    print(f"Expected number of clips: {expected_clips}") # แสดงจำนวนคลิปที่คาดว่าจะได้
    print(f"Output folder: {output_folder}") # แสดงโฟลเดอร์เอาต์พุต

    os.makedirs(output_folder, exist_ok=True) # สร้างโฟลเดอร์เอาต์พุตหากยังไม่มี

    clip_index = 0 # กำหนดค่าดัชนีคลิปเริ่มต้น
    frames_read = 0 # กำหนดค่าจำนวนเฟรมที่อ่านเริ่มต้น
    # วนลูปอ่านเฟรมและสร้างคลิป
    while frames_read < frame_count: # สร้างเส้นทางไฟล์สำหรับคลิปปัจจุบัน
        output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_clip_{clip_index + 1}.avi") # แสดงข้อมูลการสร้างคลิป
        print(f"Creating clip {clip_index + 1}: {output_file}") # แสดงข้อมูลการสร้างคลิป
        
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height)) # เริ่มต้นวิดีโอไรเตอร์สำหรับเอาต์พุต
        
        frames_written = 0 # กำหนดค่าจำนวนเฟรมที่เขียนเริ่มต้น
        for _ in range(frames_per_clip): # วนลูปเขียนเฟรมไปยังคลิป
            ret, frame = cap.read() # อ่านเฟรมจากวิดีโอ
            if ret: # ตรวจสอบว่าอ่านเฟรมสำเร็จหรือไม่
                out.write(frame) # เขียนเฟรมไปยังคลิปเอาต์พุต
                frames_written += 1 # เพิ่มจำนวนเฟรมที่เขียน
                frames_read += 1 # เพิ่มจำนวนเฟรมที่อ่าน
            else: # หากไม่สามารถอ่านเฟรมได้อีก
                break # ออกจากลูป

        out.release() # ปล่อยวิดีโอไรเตอร์

        if frames_written == 0: # หากไม่มีเฟรมถูกเขียน
            os.remove(output_file) # ลบไฟล์เอาต์พุตที่ว่างเปล่า
            break  # ออกจากลูป

        print(f"Completed clip {clip_index + 1} with {frames_written} frames") # แสดงข้อมูลการเสร็จสิ้น
        clip_index += 1 # เพิ่มดัชนีคลิป

    cap.release() # ปล่อยวิดีโอแคปเจอร์
    print(f"Completed splitting into {clip_index} clips for video: {video_path}") # แสดงข้อมูลสุดท้าย
    return clip_index # คืนค่าจำนวนคลิปที่สร้าง

def process_folders(folder_paths, clip_duration=15): # ฟังก์ชันสำหรับประมวลผลหลายโฟลเดอร์ที่มีวิดีโอ
    total_videos_processed = 0 # กำหนดค่าจำนวนวิดีโอที่ประมวลผลรวมเริ่มต้น
    total_clips_created = 0 # กำหนดค่าจำนวนคลิปที่สร้างรวมเริ่มต้น
    
    for folder in folder_paths:  # วนลูปผ่านแต่ละโฟลเดอร์
        print(f"\nProcessing folder: {folder}") # แสดงโฟลเดอร์ที่กำลังประมวลผล
        
        # แทนที่จะใช้ os.walk ใช้ os.listdir เพื่อดูแค่ไฟล์ในโฟลเดอร์นั้นๆ โดยตรง
        try:
            files = os.listdir(folder)
            video_files = [f for f in files if f.lower().endswith(('.avi', '.mp4'))] # กรองเฉพาะไฟล์วิดีโอ
            
            if not video_files: # ตรวจสอบว่าไม่มีไฟล์วิดีโอ
                print(f"No video files found in {folder}") # แสดงข้อความเตือน
                continue # ข้ามไปยังโฟลเดอร์ถัดไป
                
            print(f"Found {len(video_files)} video files:") # แสดงจำนวนไฟล์วิดีโอที่พบ
            for file in video_files:  # วนลูปผ่านไฟล์วิดีโอที่พบ
                print(f"- {file}") # แสดงชื่อไฟล์วิดีโอ
            
            # สร้างโฟลเดอร์ _splitted
            folder_name = os.path.basename(folder) # รับชื่อโฟลเดอร์
            output_folder = os.path.join(folder, f"{folder_name}_splitted") # กำหนดเส้นทางโฟลเดอร์เอาต์พุต
             # ตรวจสอบว่าโฟลเดอร์เอาต์พุตมีอยู่แล้วและมีไฟล์อยู่
            if os.path.exists(output_folder) and any(os.scandir(output_folder)): 
                print(f"Warning: Output folder exists and contains files: {output_folder}") # แสดงข้อความเตือน
                user_input = input("Do you want to process anyway? (y/n): ") # ถามผู้ใช้ยืนยันการดำเนินการ
                if user_input.lower() != 'y': # หากผู้ไม่ต้องการดำเนินการต่อ
                    print("Skipping folder...") # แสดงข้อความข้าม
                    continue # ข้ามไปยังโฟลเดอร์ถัดไป
            
            os.makedirs(output_folder, exist_ok=True) # สร้างโฟลเดอร์เอาต์พุตหากยังไม่มี
            
            for file in video_files: # วนลูปผ่านไฟล์วิดีโอที่พบ
                video_path = os.path.join(folder, file) # รับเส้นทางไฟล์ของวิดีโอ
                clips_created = split_video_into_clips(video_path, output_folder, clip_duration) # แบ่งวิดีโอเป็นคลิป
                total_clips_created += clips_created # เพิ่มจำนวนคลิปที่สร้าง
                total_videos_processed += 1 # เพิ่มจำนวนวิดีโอที่ประมวลผล
                print(f"Processed {total_videos_processed}/{len(video_files)} videos") # แสดงข้อมูลการประมวลผล

        except Exception as e: # จับข้อยกเว้นที่เกิดขึ้น
            print(f"Error processing folder {folder}: {str(e)}") # แสดงข้อความผิดพลาด
            continue # ข้ามไปยังโฟลเดอร์ถัดไป
    # แสดงสรุปผลการประมวลผล
    print(f"\nProcessing completed:") # แสดงข้อความเสร็จสิ้น
    print(f"Total videos processed: {total_videos_processed}") # แสดงจำนวนวิดีโอที่ประมวลผล
    print(f"Total clips created: {total_clips_created}") # แสดงจำนวนคลิปที่สร้าง



# ตัวอย่างการใช้งาน
folder_paths = [
    # 'D:\\HumanDetection\\GoogleMediapipe\\No_Requests\\No_Requests_resized',
    # 'D:\\HumanDetection\\GoogleMediapipe\\Person_1_Requests\\Person_1_Requests_resized',
    # 'D:\\HumanDetection\\GoogleMediapipe\\Person_2_Requests\\Person_2_Requests_resized',
    # 'D:\\HumanDetection\\GoogleMediapipe\\Person_1_and_Person_2_Requests\\Person_1_and_Person_2_Requests_resized',
    # 'D:\\HumanDetection\\GoogleMediapipe\\Turn_Off\\D_No_Requests\\D_No_Requests_resized',
    'D:\\HumanDetection\\GoogleMediapipe\\Turn_Off\\D_Person_1_and_Person_2_Requests\\D_Person_1_and_Person_2_Requests_resized',
    # 'D:\\HumanDetection\\GoogleMediapipe\\Turn_Off\\D_Person_1_Requests\\D_Person_1_Requests_resized',
    # 'D:\\HumanDetection\\GoogleMediapipe\\Turn_Off\\D_Person_2_Requests\\D_Person_2_Requests_resized'
]

process_folders(folder_paths, clip_duration=15) # เรียกใช้ฟังก์ชันประมวลผลโฟลเดอร์
 
