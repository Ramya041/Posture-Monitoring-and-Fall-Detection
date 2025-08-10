# Real-time Posture and Fall Detection System 

This project utilizes a YOLO (You Only Look Once) model for real-time human posture analysis and fall detection via a webcam feed. The system identifies standing, sitting, and falling postures, detects prolonged inactivity, and sends immediate alerts through Telegram when a fall is detected.

### Features

Real-time Posture Detection: Classifies posture into "Standing," "Sitting," or "Falling" by analyzing the aspect ratio of a person's bounding box.

Dual-Method Fall Detection: A fall is detected using two methods:


Posture Analysis: Triggered when a person's bounding box is significantly wider than it is tall.


Movement Speed: Triggered when a person's change in position between frames exceeds a set threshold.


Inactivity Alerts: Monitors individuals in "Sitting" or "Lying" postures and displays an on-screen alert if they remain inactive for a configurable duration (default is 60 seconds).


Instant Telegram Alerts: Immediately sends a notification to a predefined list of Telegram users upon detecting a fall
