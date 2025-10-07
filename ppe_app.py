import cv2
import numpy as np
import datetime
import winsound
import os
from ultralytics import YOLO

class PPEApp:
    def __init__(self):
        self.model = YOLO("bestn.pt")
        self.confidence_threshold = 0.8
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Unable to access the webcam.") 
            exit()

        cv2.namedWindow("PPE Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("PPE Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.last_saved_minute = None

    def log_detection(self, hardhat, vest, glasses, score):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open("ppe_log.txt", "a") as log_file:
            log_file.write(f"{timestamp} - Score: {score}% - Hardhat: {hardhat}, Vest: {vest}, Glasses: {glasses}\n")

    def alert_missing_ppe(self):
        winsound.Beep(1000, 500)

    def save_and_log_once_per_minute(self, frame, hardhat, vest, glasses, score):
        current_minute = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        if self.last_saved_minute != current_minute:
            if score == 100:
                folder = "full_covered_ppe"
            elif score in [25, 50]:
                folder = "half_covered_ppe"
            else:
                folder = "no_covered_ppe"

            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"missing_ppe_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(filename, frame)

            self.log_detection(hardhat, vest, glasses, score)
            self.last_saved_minute = current_minute

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame.")
            return

        frame = cv2.resize(frame, (640, 480))
        results = self.model(frame, conf=self.confidence_threshold, iou=0.3)

        hardhat_count = 0
        vest_count = 0
        glasses_count = 0

        if len(results[0].boxes) > 0:
            detections = results[0].boxes.cls
            for det in detections:
                label = results[0].names[int(det)]
                if label == 'hardhat':
                    hardhat_count += 1
                elif label == 'vest':
                    vest_count += 1
                elif label == 'safety glasses':
                    glasses_count += 1

        hardhat_status = 'Yes' if hardhat_count > 0 else 'No'
        vest_status = 'Yes' if vest_count > 0 else 'No'
        glasses_status = 'Yes' if glasses_count > 0 else 'No'

        worn_count = [hardhat_status, vest_status, glasses_status].count('Yes')
        score = worn_count * 25

        if score < 100:
            self.alert_missing_ppe()
            self.save_and_log_once_per_minute(frame, hardhat_status, vest_status, glasses_status, score)

        # Mood and poem based on score
        if score == 0:
            mood = ":c"
            poem = "No gear, no grace — danger stares you in the face."
        elif score == 25:
            mood = ":/"
            poem = "One piece won’t shield you from the storm — suit up in full form."
        elif score == 50:
            mood = ":|"
            poem = "Halfway geared is halfway safe — don’t gamble with fate."
        elif score == 100:
            mood = ":D"
            poem = "Fully armed, fully wise — safety is your greatest prize."
        else:
            mood = ":)"
            poem = "Almost there, just one to go — wear it proud and let it show."

        annotated_frame = results[0].plot()
        font_scale = 0.7
        thickness = 2

        # Overlay PPE status
        cv2.putText(annotated_frame, "Hardhat: ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(annotated_frame, hardhat_status, (150, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 255, 0) if hardhat_status == 'Yes' else (0, 0, 255), thickness)

        cv2.putText(annotated_frame, "Vest: ", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(annotated_frame, vest_status, (150, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 255, 0) if vest_status == 'Yes' else (0, 0, 255), thickness)

        cv2.putText(annotated_frame, "Safety Glasses: ", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(annotated_frame, glasses_status, (200, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 255, 0) if glasses_status == 'Yes' else (0, 0, 255), thickness)

        # Bottom-left poem block
        frame_height, frame_width = annotated_frame.shape[:2]
        poem_start_y = frame_height - 60

        cv2.putText(annotated_frame, f"Score: {score}% {mood}", (10, poem_start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        poem_lines = poem.split(" — ")
        for i, line in enumerate(poem_lines):
            y_offset = poem_start_y + 20 + (i * 20)
            cv2.putText(annotated_frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        # Timestamp at bottom-right
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text_size, _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_width = text_size[0]
        cv2.putText(annotated_frame, timestamp, (frame_width - text_width - 10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Keyboard Controls
        cv2.putText(annotated_frame, "Press [s] Save | [q] Quit", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("PPE Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("manual_save.jpg", frame)
        elif key == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            exit()

    def run(self):
        while True:
            self.update_frame()

if __name__ == '__main__':
    app = PPEApp()
    app.run() 