import cv2
from ultralytics import YOLO
import easyocr
import pytesseract
import re
import numpy as np
easyocr_reader = easyocr.Reader(['en'])

def get_vest_number_multi_ocr(img, x1, y1, x2, y2):
    """
    Try multiple OCR engines and return the best result for vest number reading
    """
    pad = 15  # Increased padding for vest detection
    y1_pad = max(0, y1 - pad)
    y2_pad = min(img.shape[0], y2 + pad)
    x1_pad = max(0, x1 - pad)
    x2_pad = min(img.shape[1], x2 + pad)
    
    vest_crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
    
    if vest_crop.size == 0:
        return None, "No valid crop"
    
    # Preprocessing for better OCR on vest numbers
    gray = cv2.cvtColor(vest_crop, cv2.COLOR_BGR2GRAY)
    
    # Multiple enhancement techniques optimized for vest text
    enhanced_images = []
    
    # Version 1: High contrast for dark text on bright vest
    contrast = cv2.convertScaleAbs(gray, alpha=2.5, beta=20)
    enhanced_images.append(("High Contrast", contrast))
    
    # Version 2: Gaussian blur + threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_images.append(("OTSU Threshold", thresh))
    
    # Version 3: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    enhanced_images.append(("Adaptive Threshold", adaptive))
    
    # Version 4: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    enhanced_images.append(("Morphological", morph))
    
    # Version 5: Invert for white text on dark background
    inverted = cv2.bitwise_not(gray)
    enhanced_images.append(("Inverted", inverted))
    
    ocr_results = []
    
    # Try each enhanced image with multiple OCR engines
    for desc, enhanced in enhanced_images:
        # Resize for better OCR
        scale_factor = 5  # Increased scale for vest text
        h, w = enhanced.shape
        resized = cv2.resize(enhanced, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
        
        # Method 1: EasyOCR
        try:
            easy_results = easyocr_reader.readtext(resized)
            for (bbox, text, confidence) in easy_results:
                if confidence > 0.2:  # Lower threshold for vest text
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(clean_text) >= 2:  # Shorter minimum length for vest numbers
                        ocr_results.append({
                            'text': clean_text,
                            'confidence': confidence,
                            'method': f'EasyOCR + {desc}',
                            'enhancement': desc
                        })
        except:
            pass
        
        # Method 2: Tesseract with different configs optimized for vest text
        tesseract_configs = [
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 13'  # Raw line for single text line
        ]
        
        for config in tesseract_configs:
            try:
                # Convert to 3-channel for tesseract
                if len(resized.shape) == 2:
                    tesseract_input = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                else:
                    tesseract_input = resized
                
                text = pytesseract.image_to_string(tesseract_input, config=config).strip()
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                if len(clean_text) >= 2:
                    # Tesseract doesn't give confidence, so we estimate based on length and characters
                    confidence = min(0.85, len(clean_text) / 6.0 + 0.4)
                    ocr_results.append({
                        'text': clean_text,
                        'confidence': confidence,
                        'method': f'Tesseract + {desc}',
                        'enhancement': desc
                    })
            except:
                pass
    
    # Return the best result automatically
    if ocr_results:
        # Sort by confidence and return the highest
        ocr_results.sort(key=lambda x: x['confidence'], reverse=True)
        best_result = ocr_results[0]
        
        # Print the results for debugging/logging purposes
        print(f"\nVest Number OCR Results:")
        print(f"Best Result: {best_result['text']} (Confidence: {best_result['confidence']:.3f}) - {best_result['method']}")
        if len(ocr_results) > 1:
            print(f"Alternative Results Found: {len(ocr_results) - 1}")
            for i, result in enumerate(ocr_results[1:4], 2):  # Show next 3 alternatives
                print(f"  {i}. {result['text']} (Confidence: {result['confidence']:.3f}) - {result['method']}")
        
        return best_result['text'], f"OCR successful with {best_result['method']}"
    
    return None, "No text detected"

# Load model
model = YOLO("model_weights.pt")
classNames = ['helmet', 'license_plate', 'no_helmet', 'no_plate', 'no_vest', 'vest']

# Read image
#img = cv2.imread("test_lisn/moto_test_one.jpg")
img = cv2.imread("test_videos/more_test.jpg")
original_height, original_width = img.shape[:2]

canvas_width = 1920
canvas_height = 1080
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 50

e
scale_w = (canvas_width * 0.8) / original_width
scale_h = (canvas_height * 0.7) / original_height
scale = min(scale_w, scale_h)

new_width = int(original_width * scale)
new_height = int(original_height * scale)
resized_img = cv2.resize(img, (new_width, new_height))

start_x = (canvas_width - new_width) // 2
start_y = int(canvas_height * 0.15)

# Run detection
results = model(img)

vest_number = None
has_vest = False
has_license_plate = False

# Process detections - look for vest and license plate
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        currentClass = classNames[cls]
        
        if currentClass == 'vest':
            has_vest = True
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Use multi-OCR function for vest number
            vest_text, status = get_vest_number_multi_ocr(img, x1, y1, x2, y2)
            
            if vest_text:
                vest_number = vest_text
                print(f"Vest number detected: {vest_number}")
            else:
                print(f"Vest OCR failed: {status}")
        
        elif currentClass == 'license_plate':
            has_license_plate = True

# Draw detections with new color scheme
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1 = int(x1 * scale)
        y1 = int(y1 * scale)
        x2 = int(x2 * scale)
        y2 = int(y2 * scale)
        
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        currentClass = classNames[cls]
        
        if currentClass == 'helmet':
            color = (0, 255, 0)  # Green
            bg_color = (0, 200, 0)
        elif currentClass == 'license_plate':
            color = (0, 255, 0)  # Green
            bg_color = (0, 200, 0)
        elif currentClass in ['vest', 'no_vest', 'no_helmet']:
            color = (0, 0, 255)  # Red
            bg_color = (0, 0, 200)
        else:  # no_plate
            color = (0, 255, 255)  # Yellow (default for other classes)
            bg_color = (0, 200, 200)
        
        box_thickness = max(2, int(scale * 3))
        cv2.rectangle(resized_img, (x1, y1), (x2, y2), color, box_thickness)
        
        label = f'{currentClass.upper().replace("_", " ")}: {conf:.0%}'
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.4, scale * 0.6)
        text_thickness = max(1, int(scale * 2))
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
        label_y = y1 - 10 if y1 > text_height + 20 else y2 + text_height + 10
        
        cv2.rectangle(resized_img, (x1, label_y - text_height - 8), 
                     (x1 + text_width + 16, label_y + 8), bg_color, -1)
        cv2.putText(resized_img, label, (x1 + 8, label_y), 
                   font, font_scale, (255, 255, 255), text_thickness)


canvas[start_y:start_y+new_height, start_x:start_x+new_width] = resized_img

# Add title based on new logic: vest + license plate detection
title = None
title_color = (0, 0, 255)  # Red for violations

if has_vest and has_license_plate:
    if vest_number:
        title = f"VIOLATION: VEST NUMBER {vest_number}"
    else:
        title = "VIOLATION: CAN'T READ VEST"

if title:
    title_font = cv2.FONT_HERSHEY_DUPLEX
    title_font_scale = 1.2
    title_thickness = 3
    
    (title_width, title_height), _ = cv2.getTextSize(title, title_font, title_font_scale, title_thickness)
    title_x = (canvas_width - title_width) // 2
    title_y = 60
    
    bg_margin = 20
    cv2.rectangle(canvas, 
                 (title_x - bg_margin, title_y - title_height - bg_margin),
                 (title_x + title_width + bg_margin, title_y + bg_margin),
                 (255, 255, 255), -1)
    
    cv2.rectangle(canvas, 
                 (title_x - bg_margin, title_y - title_height - bg_margin),
                 (title_x + title_width + bg_margin, title_y + bg_margin),
                 (0, 0, 0), 2)
    
    cv2.putText(canvas, title, (title_x, title_y), 
               title_font, title_font_scale, title_color, title_thickness)

# Add footer
footer_text = "AUTOMATED SAFETY COMPLIANCE MONITORING SYSTEM"
footer_font = cv2.FONT_HERSHEY_DUPLEX
footer_scale = 0.6
(footer_width, footer_height), _ = cv2.getTextSize(footer_text, footer_font, footer_scale, 1)
footer_x = (canvas_width - footer_width) // 2
footer_y = canvas_height - 30

cv2.putText(canvas, footer_text, (footer_x, footer_y), 
           footer_font, footer_scale, (180, 180, 180), 1)

cv2.imshow("VP Analytics Dashboard", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("vp_ready_result.jpg", canvas)
