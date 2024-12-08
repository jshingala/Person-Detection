# Import required libraries
import cv2
from ultralytics import YOLO
import os
import time
from datetime import datetime
import platform
import numpy as np

def init_camera():
    """Initialize camera with macOS specific settings"""
    camera_methods = [
        lambda: cv2.VideoCapture(0),  # Default
        lambda: cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION),  #/Users/jenilshingala/Downloads/Hackathon_Face_Recognizer(1).ipynb AVFoundation
        lambda: cv2.VideoCapture(1),  # Try second camera
        lambda: cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # Second camera with AVFoundation
    ]
    
    for method in camera_methods:
        print("Trying new camera initialization method...")
        cap = method()
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print("Successfully initialized camera")
                # Set camera properties for better quality
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
            cap.release()
    
    raise RuntimeError("Could not initialize any camera")

def calculate_iou(box1, box2):
    """Calculate intersection over union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def save_detection(frame, person_id, timestamp, crop_box=None):
    """Save both full frame and cropped person"""
    # Create directories if the y don't exists
    full_frame_dir = 'detected_persons/full_frames'
    crops_dir = 'detected_persons/crops'
    os.makedirs(full_frame_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    
    # Save full frame
    full_frame_path = f'{full_frame_dir}/person_{person_id}_{timestamp}.jpg'
    cv2.imwrite(full_frame_path, frame)
    
    # Save cropped person if crop box provided
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        crop = frame[y1:y2, x1:x2]
        crop_path = f'{crops_dir}/person_{person_id}_{timestamp}.jpg'
        cv2.imwrite(crop_path, crop)

def main():
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Operating System: {platform.system()}")
    print(f"macOS version: {platform.mac_ver()[0]}")

    # Create directory for saving detections
    if not os.path.exists('detected_persons'):
        os.makedirs('detected_persons')
    
    # Initialize YOLO11x model
    print("Loading YOLO11x model...")
    try:
        model = YOLO('yolo11n.pt')
        print("YOLO11x model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO11x model: {e}")
        print("Falling back to YOLOv8n...")
        model = YOLO('yolov11n.pt')
    
    try:
        # Initialize camera
        cap = init_camera()
        
        # Variables for statistics
        fps = 0
        frame_count = 0
        start_time = time.time()
        detection_counter = 0
        conf_threshold = 0.5  # Confidence threshold for saving detections
        person_counter = 0    # Counter for labeling persons in each frame
        total_unique_persons = 0  # Counter for total unique persons
        previous_boxes = []   # Store previous frame's detections
        tracked_persons = []  # Store coordinates of tracked persons
        
        print("\nControls:")
        print("Press 'q' to quit")
        print("Press 'up' to increase confidence threshold")
        print("Press 'down' to decrease confidence threshold")
        print("\nSaving both full frames and crops to:")
        print("- Full frames: detected_persons/full_frames/")
        print("- Crops: detected_persons/crops/")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                # Try to reinitialize camera
                cap.release()
                cap = init_camera()
                continue
            
            # Create a copy of the frame for saving (without overlays)
            clean_frame = frame.copy()
            
            # Update FPS
            frame_count += 1
            if time.time() - start_time > 1:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Reset person counter for each frame
            person_counter = 0
            current_boxes = []
            
            # Run detection with YOLO11x
            results = model.predict(
                source=frame,
                conf=0.15,        # Initial detection threshold
                show=False,
                imgsz=640,
                classes=[0]       # Person class only
            )
            
            # Process detections
            if len(results) > 0:
                # Sort detections by confidence to label more confident detections first
                boxes = results[0].boxes
                confidences = [float(box.conf[0]) for box in boxes]
                sorted_indices = sorted(range(len(confidences)), key=lambda k: confidences[k], reverse=True)
                
                for idx in sorted_indices:
                    result = boxes[idx]
                    # Get detection details
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    conf = float(result.conf[0])
                    current_box = [x1, y1, x2, y2]

                    # Check if this is a new person
                    is_new_person = True
                    for prev_box in previous_boxes:
                        if calculate_iou(current_box, prev_box) > 0.5:  # 50% overlap threshold
                            is_new_person = False
                            break
                    
                    # If new person detected with high confidence, save images
                    if is_new_person and conf > conf_threshold:
                        total_unique_persons += 1
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_detection(clean_frame, total_unique_persons, timestamp, current_box)
                        detection_counter += 1
                    
                    current_boxes.append(current_box)
                    person_counter += 1
                    
                    # Color based on confidence (green to red)
                    color = (0, int(255 * (conf / 1.0)), 0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with person number and confidence
                    label_text = f'Person {person_counter}: {conf:.2f}'
                    
                    # Create background rectangle for text
                    (text_width, text_height), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, 
                                (x1, y1 - 30), 
                                (x1 + text_width, y1), 
                                color, 
                                -1)  # Filled rectangle
                    
                    # Add white text
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show new person indicator
                    if is_new_person and conf > conf_threshold:
                        cv2.putText(frame, "New Person!", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Update previous boxes for next frame
            previous_boxes = current_boxes
            
            # Add information overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Current People: {person_counter}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Unique People: {total_unique_persons}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf Threshold: {conf_threshold:.2f}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('YOLO11x Person Detection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 82:  # Up arrow
                conf_threshold = min(conf_threshold + 0.1, 1.0)
            elif key == 84:  # Down arrow
 
