import os
import cv2
from ultralytics import YOLO
import torch
import numpy as np

# Check CUDA
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO('yolo12n.pt').to(device)

# Open video
video_path = "assets/test.mp4"
cap = cv2.VideoCapture(video_path)

# Ball trail with fading effect
ball_trail = []
max_trail_length = 20

def draw_fading_trail(frame, trail):
    """Draw trail with fading effect"""
    if len(trail) < 2:
        return
    
    for i in range(1, len(trail)):
        # Calculate alpha (transparency) based on position in trail
        alpha = i / len(trail)
        thickness = int(3 * alpha) + 1
        
        # Create colors that fade from red to yellow
        color = (0, int(255 * alpha), int(255 * (1 - alpha * 0.5)))
        
        cv2.line(frame, trail[i-1], trail[i], color, thickness)

def draw_glowing_circle(frame, center, radius, color):
    """Draw a glowing circle effect"""
    # Draw multiple circles with decreasing intensity
    for i in range(3):
        alpha_color = tuple(int(c * (0.8 - i * 0.2)) for c in color)
        cv2.circle(frame, center, radius + i * 2, alpha_color, -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame, classes=[0, 32])  # 0 = person, 32 = sports ball
    
    player_count = 0
    ball_detected = False
    
    # Process detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Count players (person class)
                if cls == 0 and conf > 0.3:  # person
                    player_count += 1
                    # Draw player bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (255, 0, 0), 2)
                
                # Track ball
                elif cls == 32 and conf > 0.4:  # sports ball
                    ball_detected = True
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    # Add to trail
                    ball_trail.append(center)
                    if len(ball_trail) > max_trail_length:
                        ball_trail.pop(0)
                    
                    # Draw glowing ball
                    draw_glowing_circle(frame, center, 6, (0, 255, 0))
                    
                    # Ball label with background
                    label = f'BALL {conf:.1f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (center[0] - label_size[0]//2 - 5, center[1] - 25),
                                (center[0] + label_size[0]//2 + 5, center[1] - 5),
                                (0, 0, 0), -1)
                    cv2.putText(frame, label, (center[0] - label_size[0]//2, center[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw fading trail
    draw_fading_trail(frame, ball_trail)
    
    # Clear trail if ball not detected for a while
    if not ball_detected and len(ball_trail) > 0:
        ball_trail = ball_trail[1:]  # Gradually remove trail
    
    # Display player count with background
    count_text = f'Players: {player_count}'
    cv2.rectangle(frame, (10, 10), (200, 50), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Enhanced Ball Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()