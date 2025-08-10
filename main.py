import cv2
import numpy as np
from ultralytics import YOLO
import time
import requests

telegram_bot_token = "7743043517:AAF6wBFy8PLLQkwJx9jTLH4Cw2BtNAIM5VA"
chat_ids = [5070281092, 1768044109,5791900798]
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    for chat_id in chat_ids:
        payload = {"chat_id": chat_id, "text": message}
        try:
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                print(f"Failed to send message to {chat_id}: {response.json()}")
        except Exception as e:
            print(f"Error sending message to {chat_id}: {e}")

class PostureFallDetection:
    def __init__(self, model_path, confidence_threshold=0.5, inactivity_duration=60, fall_speed_threshold=50):
        # Load the YOLO model
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold  # Confidence threshold for detection
        self.inactivity_duration = inactivity_duration  # Duration for inactivity alert (in seconds)
        self.fall_speed_threshold = fall_speed_threshold  # Speed threshold to detect fall (in pixels per frame)
        self.last_sitting_time = {}  # Dictionary to track inactivity time for each person
        self.previous_positions = {}  # To track the previous position of the person


    def analyze_posture(self, x1, y1, x2, y2):
        """
        Analyze posture based on bounding box dimensions.
        :param x1, y1, x2, y2: Coordinates of the bounding box
        :return: Posture label (e.g., "Standing", "Sitting", "Lying") and fall detection flag
        """
        box_width = x2 - x1
        box_height = y2 - y1
        aspect_ratio = box_width / max(box_height, 1)

        if aspect_ratio < 0.75:  # Taller than wide
            return "Standing", False
        elif 0.75 <= aspect_ratio <= 1.5:  # Nearly square
            return "Sitting", False
        else:  # Wider than tall
            send_telegram_alert("🚨 Fall detected! Immediate action required!")
            return "Falling", False  # "Lying" posture (no fall detection yet)

    def calculate_speed(self, person_id, x1, y1):
        """
        Calculate speed (distance traveled per frame) based on previous position.
        :param person_id: Unique identifier for the person based on bounding box coordinates
        :param x1, y1: Current bounding box coordinates
        :return: Speed (distance per frame)
        """
        if person_id in self.previous_positions:
            prev_x, prev_y = self.previous_positions[person_id]
            distance = np.sqrt((x1 - prev_x) ** 2 + (y1 - prev_y) ** 2)
            self.previous_positions[person_id] = (x1, y1)
            return distance
        else:
            # First frame for this person, no speed calculation yet
            self.previous_positions[person_id] = (x1, y1)
            return 0

    def detect_postures_and_falls(self, frame):
        """
        Perform object detection, posture analysis, and fall detection.
        :param frame: The video frame to process
        :return: Annotated frame and fall detection flag
        """
        results = self.model(frame)
        falls_detected = False
        current_time = time.time()  # Get the current time
        fall_alert_triggered = False

        for result in results[0].boxes:
            cls = result.cls.item()  # Class index
            label = self.model.names[int(cls)]  # Class name
            conf = result.conf.item()  # Confidence score

            # Only process "person" detections with a high confidence score
            if label == "person" and conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, result.xyxy[0][:4])  # Extract bounding box coordinates
                posture, is_fall = self.analyze_posture(x1, y1, x2, y2)

                # Calculate speed to detect fall (speed threshold for fall)
                person_id = f"{x1}-{y1}-{x2}-{y2}"  # Unique ID based on bounding box coordinates
                speed = self.calculate_speed(person_id, x1, y1)

                if speed > self.fall_speed_threshold:
                    is_fall = True
                    falls_detected = True
                    # Sound alert for fall detection
                    if not fall_alert_triggered:
                        winsound.Beep(1000, 500)  # 1000 Hz for 500 ms
                        fall_alert_triggered = True
                else:
                    fall_alert_triggered = False  # Reset fall alert once speed is under threshold

                # Track sitting/lying postures to detect inactivity
                if posture in ["Sitting", "Lying"]:
                    if person_id not in self.last_sitting_time:
                        self.last_sitting_time[person_id] = current_time  # Start the timer for this person
                    elif current_time - self.last_sitting_time[person_id] > self.inactivity_duration:
                        # Raise an alert if this person has been sitting or lying for over 1 minute
                        cv2.putText(
                            frame,
                            f"Error: Person {person_id} hasn't moved for 1 minute!",
                            (x1, y1 - 60),  # Place the alert above the box
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),  # Red color for the alert
                            2,
                        )
                else:
                    # Reset the timer for this person if they are standing
                    if person_id in self.last_sitting_time:
                        del self.last_sitting_time[person_id]

                # Draw bounding box and label
                color = (0, 0, 255) if is_fall else (0, 255, 0)  # Red for fall, green otherwise
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Adjust text position to avoid overlap
                label_position = (x1, y1 - 10)  # Adjust text position slightly above the bounding box

                # Display the posture and confidence on separate lines to avoid overlap
                cv2.putText(
                    frame,
                    f"Posture: {posture}",
                    (x1, y1 - 30),  # Adjust y-position further above
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {conf:.2f}",
                    (x1, y1 - 10),  # Position confidence below posture
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        return results[0].plot(), falls_detected


# Initialize posture and fall detection system
model_path = "yolo11n-pose.pt"  # Replace with the path to your trained YOLO model
detector = PostureFallDetection(model_path)

# Open webcam
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

    # Detect postures and falls
    annotated_frame, falls_detected = detector.detect_postures_and_falls(frame)

    # Show the processed frame
    cv2.imshow("Posture and Fall Detection", annotated_frame)

    # Trigger alert if a fall is detected (with sound)
    if falls_detected:
        print("Fall detected! Triggering sound alert...")

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
