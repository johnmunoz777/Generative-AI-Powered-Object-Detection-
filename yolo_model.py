import numpy as np
from ultralytics import YOLO
import cv2
# Initialize video capture
cap = cv2.VideoCapture("test_videos/peru_test_video.mp4")
# Load your custom model
model = YOLO(r"C:\Users\john\yolo_peru_project\model_weights.pt")
# Class names
classNames = ['helmet', 'license_plate', 'no_helmet', 'no_plate', 'no_vest', 'vest']
SKIP_FRAMES = 2  
RESIZE_SCALE = 0.3  
CONF_THRESHOLD = 0.5  
frame_count = 0
last_results = []
font = cv2.FONT_HERSHEY_DUPLEX  
font_scale = 0.9
thickness = 2
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
while True:
    success, img = cap.read()
    if not success:
        break
    
    if frame_count % SKIP_FRAMES == 0:
        height, width = img.shape[:2]
        small_width = int(width * RESIZE_SCALE)
        small_height = int(height * RESIZE_SCALE)
        small_img = cv2.resize(img, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        results = model(small_img, stream=True, conf=CONF_THRESHOLD, verbose=False, max_det=20)
        last_results = []
        scale_factor = 1.0 / RESIZE_SCALE
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Scale coordinates back up
                    x1, y1, x2, y2 = box.xyxy[0] * scale_factor
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    last_results.append((x1, y1, x2, y2, conf, cls))
    
    for x1, y1, x2, y2, conf, cls in last_results:
        currentClass = classNames[cls]
        if currentClass in ['no_helmet', 'no_vest', 'no_plate']:
            box_color = (0, 0, 255)  # BRIGHT RED for violations
            bg_color = (0, 0, 180)   # Darker red for text background
        else:
            box_color = (0, 255, 0)  # BRIGHT GREEN for compliance
            bg_color = (0, 180, 0)   # Darker green for text background
        
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 4)
        
        corner_len = 25
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), box_color, 4)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), box_color, 4)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), box_color, 4)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), box_color, 4)
        label = f'{currentClass.upper()}: {conf:.0%}'
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = y1 - 10 if y1 > 40 else y2 + text_height + 10
        cv2.rectangle(img, 
                     (x1 - 2, label_y - text_height - 8), 
                     (x1 + text_width + 10, label_y + 5),
                     bg_color, -1)
        
        cv2.putText(img, label, (x1 + 5, label_y),
                   font, font_scale, (255, 255, 255), thickness)
        
        
        cv2.circle(img, (x1 + 15, y1 + 15), 10, (255, 255, 255), -1)  
        cv2.circle(img, (x1 + 15, y1 + 15), 8, box_color, -1)  
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()