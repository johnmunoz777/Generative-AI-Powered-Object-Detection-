import numpy as np
from ultralytics import YOLO
import cv2
import cvzone


def run_ultra_fast():
    # Video setup
    cap = cv2.VideoCapture("C:/Users/john/yolo_peru_project/test_videos/peru_test.mp4")
    # Use NANO model for maximum speed
    model_yolo = YOLO("yolov8n.pt")  # Downloads automatically if not present
    model_custom = YOLO("C:/Users/john/yolo_peru_project/model_weights.pt")  #
    classNames_custom = ['helmet', 'license_plate', 'no_helmet', 'no_plate', 'no_vest', 'vest']
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    
    # SPEED SETTINGS - ADJUST THESE FOR MAX PERFORMANCE
    SKIP_FRAMES = 5  # Only check for motorbikes every 5 frames
    SCALE_FACTOR = 0.3  # Aggressive downscaling for super fast detection
    DETECTION_MEMORY = 15  # Remember motorbike for 15 frames
    CONF_THRESHOLD = 0.4  # Higher threshold = fewer detections = faster
    
    frame_count = 0
    memory_counter = 0
    last_motorbike_detected = False
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Only check for motorbikes every SKIP_FRAMES
        if frame_count % SKIP_FRAMES == 0:
            # MAXIMUM SPEED: Tiny image for detection
            tiny_img = cv2.resize(img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            # ONLY detect motorbikes (class 3), nothing else
            results = model_yolo(tiny_img, stream=True, classes=[3], conf=CONF_THRESHOLD, verbose=False)
            
            motorbike_found = False
            for r in results:
                if len(r.boxes) > 0:
                    motorbike_found = True
                    # Draw one simple box for visual feedback (optional - remove for more speed)
                    for box in r.boxes[:1]:  # Only draw first motorbike
                        x1, y1, x2, y2 = map(lambda x: int(x/SCALE_FACTOR), box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, "BIKE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break
            
            if motorbike_found:
                last_motorbike_detected = True
                memory_counter = DETECTION_MEMORY
        
        # Check memory
        if memory_counter > 0:
            memory_counter -= 1
            last_motorbike_detected = True
        else:
            last_motorbike_detected = False
        
        # Only run custom model if motorbike detected
        if last_motorbike_detected:
         
            results_custom = model_custom(img, stream=True, conf=0.4, verbose=False)
            
            for r in results_custom:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    currentClass = classNames_custom[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Red for violations (no_helmet, no_vest), Green for compliance
                    if currentClass in ['no_helmet', 'no_vest', 'no_plate']:
                        color = (0, 0, 255)  # RED for violations
                        bg_color = (0, 0, 200)  # Darker red for background
                    else:
                        color = (0, 255, 0)  # GREEN for compliance
                        bg_color = (0, 200, 0)  # Darker green for background
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)  # Thicker box (3 pixels)
                    
                    # Prepare label with confidence
                    label = f"{currentClass} {conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8  # Bigger font
                    thickness = 2
                    
                    # Get text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    cv2.rectangle(img, 
                                 (x1, y1 - text_height - 10), 
                                 (x1 + text_width + 10, y1), 
                                 bg_color, 
                                 -1)  # -1 means filled
                    
                    # Draw white text on colored background for maximum contrast
                    cv2.putText(img, label, (x1 + 5, y1 - 5), 
                              font, font_scale, (255, 255, 255), thickness)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    run_ultra_fast()
    
    
 