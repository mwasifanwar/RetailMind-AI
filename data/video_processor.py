# data/video_processor.py
import cv2
import numpy as np
from datetime import datetime
import os
from collections import deque

class VideoProcessor:
    def __init__(self, source=0, frame_skip=5):
        self.source = source
        self.frame_skip = frame_skip
        self.cap = None
        self.frame_count = 0
        self.processed_frames = deque(maxlen=100)
        
    def initialize_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            return self.cap.isOpened()
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False
    
    def process_video_stream(self, callback_function, duration=None):
        if not self.initialize_camera():
            return False
        
        start_time = datetime.now()
        
        try:
            while True:
                if duration and (datetime.now() - start_time).seconds >= duration:
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                processed_frame = self._preprocess_frame(frame)
                
                results = callback_function(processed_frame, self.frame_count)
                
                self.processed_frames.append({
                    'frame': processed_frame,
                    'results': results,
                    'timestamp': datetime.now()
                })
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.release_camera()
        
        return True
    
    def _preprocess_frame(self, frame):
        processed_frame = cv2.resize(frame, (640, 480))
        
        processed_frame = cv2.GaussianBlur(processed_frame, (3, 3), 0)
        
        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=1.2, beta=10)
        
        return processed_frame
    
    def process_video_file(self, file_path, callback_function):
        if not os.path.exists(file_path):
            return False
        
        self.cap = cv2.VideoCapture(file_path)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                processed_frame = self._preprocess_frame(frame)
                
                results = callback_function(processed_frame, self.frame_count)
                
                self.processed_frames.append({
                    'frame': processed_frame,
                    'results': results,
                    'timestamp': datetime.now()
                })
        
        finally:
            self.release_camera()
        
        return True
    
    def release_camera(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def extract_frames_for_training(self, video_path, output_dir, frames_per_second=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frames_per_second)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return saved_count
    
    def create_heatmap_video(self, heatmap_data, output_path, original_video_path=None):
        if original_video_path:
            cap = cv2.VideoCapture(original_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            fps = 30
            width, height = 640, 480
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for heatmap in heatmap_data:
            heatmap_resized = cv2.resize(heatmap, (width, height))
            
            heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
            out.write(heatmap_colored)
        
        out.release()
        return output_path
    
    def get_processing_stats(self):
        return {
            'total_frames_processed': self.frame_count,
            'frames_in_buffer': len(self.processed_frames),
            'processing_rate': self.frame_count / max(1, (datetime.now() - getattr(self, 'start_time', datetime.now())).seconds)
        }