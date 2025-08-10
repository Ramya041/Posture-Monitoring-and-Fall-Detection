import cv2
import numpy as np
from ultralytics import YOLO
import time
import requests
import os
from dotenv import load_dotenv
import winsound  # Windows sound alert (optional)

# Load environment variables
load_dotenv()

# Get token and chat IDs from .env
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_ids = os.getenv("TELEGRAM_CHAT_IDS").split(",")  # Convert comma-separated string to list

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    for chat_id in chat_ids:
        payload = {"chat_id": chat_id.strip(), "text": message}
        try:
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                print(f"Failed to send message to {chat_id}: {response.json()}")
        except Exception as e:
            print(f"Error sending message to {chat_id}: {e}")

class PostureFallDetection:
    def __init__(self, model_path, confidence_threshold=0.5, inactivity_duration=60, fall_speed_threshold=50):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.inactivity_duration = inactivity_duration
        self.fall_speed_threshold = fall_speed_threshold
        self.last_sitting_time = {}
        self.previous_positions = {}

    def analyze_posture(self, x1, y1, x2, y2):
        box_width = x2 - x1
        box_height = y2 - y1
        aspect_ratio = box_width / max(box_height, 1)

        if aspect_ratio < 0.75:
            return "Standing", False
        elif 0.75 <= aspect_ratio <= 1.5:
            return "Sitting", False
        else:
            send_telegram_alert("🚨 Fall detected! Immediate action required!")
            return "Falling", False

    def calculate_speed(self, person_id, x1, y1):
        if person_id in self.previous_positions:
            prev_x, prev_y = self.previous_positions[person_id]
            distance = np.sqrt((x1 - prev_x) ** 2 + (y1 - prev_y) ** 2)
            self.previous_positions[person_id] = (x1, y1)
            return distance
        else:
            self.previous_positions[person_id] = (x1, y1)
            return 0

    def detect_postures_and_falls(self, frame):
        results = self.model(frame)
        falls_detected = False
        current_time = time.time()
        fall_alert_triggered = False

        for result in results[0].boxes:
            cls = result.cls.item()
            label = self.model.names[int(cls)]
            conf = result.conf.item()

            if label == "person" and conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, result.xyxy[0][:4])
                posture, is_fall = self.analyze_posture(x1, y1, x2, y2)

                person_id = f"{x1}-{y1}-{x2}-{y2}"
                speed = self.calculate_speed(person_id, x1, y1)

                if speed > self.fall_speed_threshold:
                    is_fall = True
                    falls_detected = True
                    if not fall_alert_triggered:
                        winsound.Beep(1000, 500)
                        fall_alert_triggered = True
                else:
                    fall_alert_triggered = False

                if posture in ["Sitting", "Lying"]:
                    if person_id not in self.last_sitting_time:
                        self.last_sitting_time[person_id] = current_time
                    elif current_time - self.last_sitting_time[person_id] > self.inactivity_duration:
                        cv2.putText(
                            frame,
                            f"Error: Person {person_id} hasn't moved for 1 minute!",
                            (x1, y1 - 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                else:
                    if person_id in self.last_sitting_time:
                        del self.last_sitting_time[person_id]

                color = (0, 0, 255) if is_fall else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(frame, f"Posture: {posture}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Confidence: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return results[0].plot(), falls_detected

# Run Detection
if __name__ == "__main__":
    model_path = "yolo11n-pose.pt"
    detector = PostureFallDetection(model_path)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        exit()

    print("Starting posture and fall detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        annotated_frame, falls_detected = detector.detect_postures_and_falls(frame)

        cv2.imshow("Posture and Fall Detection", annotated_frame)

        if falls_detected:
            print("Fall detected! Triggering sound alert...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
