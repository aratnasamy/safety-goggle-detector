import cv2
import numpy as np
import datetime
import os
import threading
import time
import subprocess
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
INFERENCE_INTERVAL = 0  # How often to run detection (in seconds)
FACE_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for face detection (0.0 to 1.0)
GLASSES_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for glasses detection (0.0 to 1.0)
DEFAULT_GOGGLES_NAMES = [ # Labels that count as goggles/eye protection
    "goggles",
    "safety goggles",
    "safety glasses",
    "safety-glasses",
    "eye protection",
    "eye_protection",
    "eyewear",
    "protective eyewear",
    "protective_eyewear",
]
# ============================================================

# Cross-platform audio alert
try:
    import winsound
    AUDIO_AVAILABLE = 'windows'
except ImportError:
    AUDIO_AVAILABLE = 'unix'

def play_alert_tone():
    """Play an alert tone based on the operating system."""
    if AUDIO_AVAILABLE == 'windows':
        winsound.Beep(1000, 500)
    elif AUDIO_AVAILABLE == 'unix':
        # Play sound file from project sounds folder (truly async)
        sound_file = os.path.join(os.path.dirname(__file__), 'sounds', 'alert.wav')
        if os.path.exists(sound_file):
            subprocess.Popen(['afplay', sound_file], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        else:
            # Fallback to system sound if project sound doesn't exist
            subprocess.Popen(['afplay', '/System/Library/Sounds/Ping.aiff'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)

class SafetyGoggleDetector:
    def __init__(self):
        # Load YOLO model for safety glasses
        self.model = YOLO("best.pt")
        
        # Load face detection model (OpenCV DNN)
        print("Loading face detection model...")
        self.face_net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Unable to access the webcam.") 
            exit()
        
        # Optimize camera settings for better performance - uncomment if needed
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame

        # Create a resizable window (removed fullscreen for better macOS compatibility)
        cv2.namedWindow("Safety Goggle Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Safety Goggle Detection", 960, 720)

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        
        # Cached results (thread-safe)
        self.results_lock = threading.Lock()
        self.cached_face_results = []  # List of (face_box, has_glasses, glasses_boxes)
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def alert_missing_safety_goggles(self):
        """Trigger alert sound for missing safety goggles."""
        play_alert_tone()
    
    def _inference_loop(self):
        """Background thread that runs inference periodically."""
        while self.running:
            cycle_start = time.time()
            
            # Get the latest frame
            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.1)
                    continue
            frame = self.latest_frame.copy()
            
            # Step 1: Detect faces
            face_start = time.time()
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            face_detections = self.face_net.forward()
            face_time = time.time() - face_start
            
            face_results = []
            yolo_times = []
            
            # Step 2: For each detected face, check for safety glasses
            for i in range(face_detections.shape[2]):
                confidence = face_detections[0, 0, i, 2]
                
                if confidence > FACE_CONFIDENCE_THRESHOLD:
                    # Get face bounding box
                    box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    # Ensure box is within frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Crop face region
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Run YOLO inference on face region
                    yolo_start = time.time()
                    results = self.model(face_crop, conf=GLASSES_CONFIDENCE_THRESHOLD, iou=0.3)
                    yolo_times.append(time.time() - yolo_start)
                    
                    has_glasses = False
                    glasses_boxes = []
                    
                    if len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            label = results[0].names[int(box.cls)]
                            if label in DEFAULT_GOGGLES_NAMES:
                                has_glasses = True
                                # Convert box coordinates from crop to full frame
                                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                # Adjust coordinates back to full frame
                                glasses_boxes.append((x1 + bx1, y1 + by1, x1 + bx2, y1 + by2, conf))
                    
                    face_results.append({
                        'face_box': (x1, y1, x2, y2),
                        'has_glasses': has_glasses,
                        'glasses_boxes': glasses_boxes
                    })
                    
                    if not has_glasses:
                        self.alert_missing_safety_goggles()
            
            # Timing report
            cycle_time = time.time() - cycle_start
            total_yolo_time = sum(yolo_times)
            print(f"Face detection: {face_time*1000:.1f}ms | YOLO ({len(yolo_times)} faces): {total_yolo_time*1000:.1f}ms | Total: {cycle_time*1000:.1f}ms | FPS: {1/cycle_time:.1f}")
            
            # Log detection every minute
            if face_results:
                total_faces = len(face_results)
                faces_with_glasses = sum(1 for f in face_results if f['has_glasses'])
                
                current_minute = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            
            # Update cached results (thread-safe)
            with self.results_lock:
                self.cached_face_results = face_results
            
            # Wait before next inference
            time.sleep(INFERENCE_INTERVAL)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame.")
            return
        
        # Share frame with inference thread
        with self.frame_lock:
            self.latest_frame = frame
        
        # Get cached results (thread-safe)
        with self.results_lock:
            face_results = self.cached_face_results.copy()
        
        # Draw results
        annotated_frame = frame.copy()
        
        total_faces = len(face_results)
        faces_with_glasses = 0
        
        for result in face_results:
            face_box = result['face_box']
            has_glasses = result['has_glasses']
            glasses_boxes = result['glasses_boxes']
            
            if has_glasses:
                faces_with_glasses += 1
            
            # Draw face box (green if has glasses, red if not)
            x1, y1, x2, y2 = face_box
            color = (0, 255, 0) if has_glasses else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw status on face box
            status = "Goggles OK" if has_glasses else "NO GOGGLES!"
            cv2.putText(annotated_frame, status, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw safety glasses bounding boxes
            for gx1, gy1, gx2, gy2, conf in glasses_boxes:
                cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"glasses {conf:.2f}", (gx1, gy1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Overlay detection status
        cv2.putText(annotated_frame, f"Faces Detected: {total_faces}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"With Goggles: {faces_with_glasses}/{total_faces}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if faces_with_glasses == total_faces else (0, 0, 255), 2)
        
        # Keyboard control hint
        cv2.putText(annotated_frame, "Press [q] to Quit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Safety Goggle Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            self.running = False  # Stop inference thread
            self.inference_thread.join(timeout=2)  # Wait for thread to finish
            self.cap.release()
            cv2.destroyAllWindows()
            exit()

    def run(self):
        while True:
            self.update_frame()

if __name__ == '__main__':
    app = SafetyGoggleDetector()
    app.run()