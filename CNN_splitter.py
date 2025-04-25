import os
import cv2

def split_video_into_clips(video_path, output_folder, clip_duration=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames_per_clip = clip_duration * fps

    expected_clips = (frame_count + frames_per_clip - 1) // frames_per_clip

    print(f"\nProcessing video: {video_path}")
    print(f"FPS: {fps}, Total Frames: {frame_count}, Resolution: {width}x{height}")
    print(f"Expected number of clips: {expected_clips}")
    print(f"Output folder: {output_folder}")

    os.makedirs(output_folder, exist_ok=True)

    clip_index = 0
    frames_read = 0

    while frames_read < frame_count:
        output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_clip_{clip_index + 1}.avi")
        print(f"Creating clip {clip_index + 1}: {output_file}")
        
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        frames_written = 0
        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                frames_written += 1
                frames_read += 1
            else:
                break

        out.release()

        if frames_written == 0:
            os.remove(output_file)
            break

        print(f"Completed clip {clip_index + 1} with {frames_written} frames")
        clip_index += 1

    cap.release()
    print(f"Completed splitting into {clip_index} clips for video: {video_path}")
    return clip_index

def process_folders(folder_paths, clip_duration=15):
    total_videos_processed = 0
    total_clips_created = 0
    
    for folder in folder_paths:
        print(f"\nProcessing folder: {folder}")
        
        # แทนที่จะใช้ os.walk ใช้ os.listdir เพื่อดูแค่ไฟล์ในโฟลเดอร์นั้นๆ โดยตรง
        try:
            files = os.listdir(folder)
            video_files = [f for f in files if f.lower().endswith(('.avi', '.mp4'))]
            
            if not video_files:
                print(f"No video files found in {folder}")
                continue
                
            print(f"Found {len(video_files)} video files:")
            for file in video_files:
                print(f"- {file}")
            
            # สร้างโฟลเดอร์ _splitted
            folder_name = os.path.basename(folder)
            output_folder = os.path.join(folder, f"{folder_name}_splitted")
            
            if os.path.exists(output_folder) and any(os.scandir(output_folder)):
                print(f"Warning: Output folder exists and contains files: {output_folder}")
                user_input = input("Do you want to process anyway? (y/n): ")
                if user_input.lower() != 'y':
                    print("Skipping folder...")
                    continue
            
            os.makedirs(output_folder, exist_ok=True)
            
            for file in video_files:
                video_path = os.path.join(folder, file)
                clips_created = split_video_into_clips(video_path, output_folder, clip_duration)
                total_clips_created += clips_created
                total_videos_processed += 1
                print(f"Processed {total_videos_processed}/{len(video_files)} videos")

        except Exception as e:
            print(f"Error processing folder {folder}: {str(e)}")
            continue

    print(f"\nProcessing completed:")
    print(f"Total videos processed: {total_videos_processed}")
    print(f"Total clips created: {total_clips_created}")



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

process_folders(folder_paths, clip_duration=15)
