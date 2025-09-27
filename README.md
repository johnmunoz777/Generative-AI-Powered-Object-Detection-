# Synthetic to Street: Generative AI-Powered Object Detection

**Research Implementation for AIPR 2025**  
**Contributors:** John Munoz MSDS, Tom Bukowski MSDS, Dr. Mike Busch PhD

---

## Executive Summary

This repository contains the implementation of an Motorcycle Compliance Object Detection Model. <br>
We developed an automated compliance monitoring system capable of real-time detection of safety violations. <br>
We utilized real-world data as well as synthetic video data generated with Google’s Veo 2 and Veo 3 models  <br>


### Key Achievement
- **81% mAP50** accuracy achieved through hybrid synthetic-real data approach
- **86% precision** in detecting safety compliance violations
- **Real-time processing** capability for traffic monitoring systems

---

[![Model Results](example_gifs/model_results.gif)](example_gifs/model_results.gif)
# Problem Statement

In April of 2025, Peru passed a new law requiring all motorcycle operators must wear certified helmets and reflective vests displaying license plate numbers.
Manual enforcement presents significant challenges:

- Resource-intensive manual inspections
- Limited scalability for nationwide coverage
- Inconsistent enforcement across regions
- Delayed response to violations

### Our Solution
An automated detection system utilizing synthetic data generation to overcome data scarcity challenges while maintaining high accuracy in real-world deployments.

[![Model Results_two](example_gifs/dash_exseven.gif)](example_gifs/dash_exseven.gif)

---
## Methodology
### Data Generation Pipeline

We developed three progressive models, each building upon lessons learned from previous iterations:

#### Model 1: Text-to-Video Generation
- **148 synthetic videos** generated via text prompts
- **2,220 frames** extracted (15 frames/video)
- **Resolution:** 1280x720 pixels
- **Limitation:** Insufficient visual diversity

[![Model Resul](example_gifs/c.gif)](example_gifs/c.gif) <br>
[![Model Resu](example_gifs/e.gif)](example_gifs/e.gif)


#### Model 2: Image-Enhanced Generation
- **Base:** 2,220 frames from Model 1
- **Enhancement:** 100 videos via image-to-video prompting
- **Total Dataset:** 3,210 images
- **Improvement:** Greater scene diversity and realism

<p align="center">
  <img src="example_gifs/starter_image.jpg" alt="Starter Image" width="45%"/>
  <img src="example_gifs/model_two.gif" alt="Model Two GIF" width="45%"/>
</p>


<p align="center">
  <img src="example_gifs/Picture2.jpg" alt="Starter Imag" width="45%"/>
  <img src="example_gifs/b.gif" alt="Model Two GIF" width="45%"/>
</p>


#### Model 3: Hybrid Approach
- **Synthetic Data:** 3,210 images from Models 1&2
- **Real-World Data:** 512 authentic images with augmentation
- **Final Dataset:** 4,234 images
- **Result:** Superior generalization capability

![Model 3 Generation](images/placeholder_model3_generation.gif)

### Class Distribution Analysis

| Class | Model 1 | Model 2 | Model 3 | Balance Status |
|-------|---------|---------|---------|----------------|
| 🪖 Helmet | 1,842 | 2,456 | 2,658 | Well-represented |
| ❌ No Helmet | 1,523 | 2,012 | 2,154 | Well-represented |
| 🦺 Vest | 1,785 | 2,389 | 2,877 | Well-represented |
| ⚠️ No Vest | 2,102 | 2,654 | 3,054 | Well-represented |
| 🔢 License Plate | 1,456 | 1,923 | 2,341 | Well-represented |
| 🚫 No Plate | 512 | 689 | 774 |  Under-represented |

## Technical Architecture

### YOLOv8 Configuration
```yaml
# Model Specifications
architecture: YOLOv8l (Large variant)
pretrained_weights: COCO dataset
input_resolution: 640x640
optimizer: AdamW
learning_rate: 0.001
batch_size: 16
augmentation: True
Platform: Google Colab Pro+
GPU: NVIDIA A100 (40GB VRAM)
Training Infrastructure
Epochs: 50 (Models 1&2), 100 (Model 3)
```
## Model Evaluation Results
| Model | Dataset                          | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|-------|----------------------------------|---------------|------------|-------|----------|
| 1     | Text-to-Prompt (Synthetic only)  | 0.675         | 0.513      | 0.352 | 0.174    |
| 2     | Image-to-Prompt (Synthetic + extra) | 0.737      | 0.635      | 0.439 | 0.207    |
| 3     | Hybrid (Synthetic + Real images) | 0.860         | 0.799      | 0.708 | 0.398    |
<br>


##  Class-Level Performance Improvements

| Class        | Model 1 (Text) | Model 2 (Text + Image) | Model 3 (Hybrid) | Key Observations |
|--------------|----------------|-------------------------|------------------|------------------|
| **Helmet**   | mAP50: 0.52 | mAP50: 0.90  | mAP50: 0.92  | Huge jump from Model 2; near-perfect in Model 3. |
| **Vest**     | Very low     | Moderate ≈0.70       | 0.91 / 0.87    | Real data in Model 3 enabled strong generalization. |
| **License Plate** | 0.64  | ≈0.85                 | 0.89 / 0.86    | Solid across all; incremental lift in Model 3. |
| **No Plate** | 0.00       | 0.36                  | 0.45           | Failure in Model 1; only detectable in 2+; Model 3 shows progress. |
| **No Vest**  | Weak       | Low–Moderate         | <0.50          | Still challenging; Model 3 best but below 0.50 mAP. |
| **No Helmet**| Weak         | Moderate             | ≈0.80+         | Major lift in Model 3; strong generalization. |

<br>




<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <strong>YOLO Detection Results</strong><br>
      <img src="pictures_for_read_me/yolo_results.jpg" alt="YOLO Detection Results" width="100%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <strong>Validation Results</strong><br>
      <div style="margin-top:20px;">
        <img src="pictures_for_read_me/validation_results.jpg" alt="Validation Results" width="100%"/>
      </div>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%" valign="top">
      <strong>Confusion Matrix</strong><br>
      <img src="pictures_for_read_me/confusion_matrix_results.jpg" alt="Confusion Matrix" width="100%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <strong>Precision–Recall Curve</strong><br>
      <div style="margin-top:20px;">
        <img src="pictures_for_read_me/precision_recall_results.jpg" alt="Precision–Recall Curve" width="100%"/>
      </div>
    </td>
  </tr>
</table>



## Project Structure  

###  Dropbox Exports  
- **Model 1 Text-To-Prompt Video Results/** – Synthetic videos generated using text-to-prompt.  
- **Model 2 Videos Text-To-Image Result/** – Synthetic videos generated using image-to-prompt.  

###  YOLO Model Results  
- **all_model_results.py** – Script to build and evaluate YOLO models.  
- **model-weights.pt** – Trained YOLO weights (from Model 3).  

### Google Drive Datasets  
- **Model 1/** – Images and labels from text-to-prompt.  
- **Model 2/** – Images and labels from text + image-to-prompt.  
- **Model 3/** – Hybrid dataset (synthetic + real-world images).  
- **data.yaml** – YOLO dataset configuration (classes + train/val/test).  

### Pre-Processing  
- **extract_frames.py** – Extracts frames from videos.  
- **split_images.py** – Splits data into train/val/test sets.  
- **image_augmentation.py** – Applies augmentations (scaling, brightness, contrast, etc.).  

### Object Detection on Videos  
- **yolo_model.py** – YOLO inference on custom dataset.  
- **yolo_model_with_coco.py** – YOLO inference using COCO classes for comparison.  

###  Dashboard  
- **finished_dashboard.py** – Streamlit dashboard for visualizing model performance.  

###  Test Examples  
- **test_examples/** – Example images and files used in dashboard + OCR tests.  



##  Data Resources  

### 🎬 Video Results  
- [Model 1 Videos – Dropbox Repository](https://www.dropbox.com/scl/fo/6jhvwh6lq166buio6o4vi/AEEvkHp8TXHU-5OpbtYprX0?rlkey=jvksmrftng7v1hwn3zjlrjxht&st=9fodicvg&dl=0)  
- [Model 2 Videos – Dropbox Repository](https://www.dropbox.com/scl/fo/d743mq5qja5i506v5owju/AJOalYCvmGseXKVF8ZBt3PY?rlkey=v5rj13na571utijm0cvijfnr4&st=oto20wxa&dl=0)  

### 📸 Training Datasets  
- [Model 1 Dataset – Google Drive](https://drive.google.com/drive/folders/1RspvxtjfR8ARrki6Z4j6bzJEoz6eq_Qs?usp=drive_link)  
- [Model 2 Dataset – Google Drive](https://drive.google.com/drive/folders/1kjRnt2KX4i3tZgRLsVXuQiApVCQrmu8b?usp=drive_link)  
- [Model 3 Dataset – Google Drive](https://drive.google.com/drive/folders/17U8tqVlf9sJ-J8u9K8fccjStmTAsFP_4?usp=drive_link)  

---

 **Supporting Script**  
- **all_model_results.py** – Results of building YOLO models in Colab.  



##  Hugging Face Dashboard – Compliance Detector  
[ Live Demo on Hugging Face](https://huggingface.co/spaces/johngmunoz/Motorcycle)

The Compliance Detector is an interactive Streamlit dashboard deployed on Hugging Face Spaces. It provides multiple modules for video and image-based motorcycle compliance detection.

---

### 🎥 Video Detections  
- **Upload a Video** – Supports MP4, MOV, AVI, MKV (up to 200MB).  
- **Models Available:**  
  - `yolo_custom` – Custom YOLOv8 model (Model 3: Hybrid synthetic + real).  
  - `coco + custom` – First detects motorcycles with COCO, then applies custom compliance model.  
- **Controls:**  
  - **Skip Frames** – Speed vs accuracy trade-off.  
  - **Resize Scale** – Adjusts video input size for faster inference.  
  - **Confidence Slider** – Minimum confidence threshold for detection.  
- **Output:**  
  - Bounding box detections for helmets, vests, license plates, and violations.  
  - Live video preview with adjustable performance settings.  
![Model Resul](example_gifs/final_model.gif)

---

###  Image Detections  
- **Upload an Image** – Supports JPG and PNG.  
- **Custom model (.pt)** – Loads trained YOLO model weights.  
- **OCR Options:**  
  1. **License Plate OCR** – Detects riders missing a vest and extracts plate text.  
  2. **Vest OCR** – Detects vest violations and extracts vest number (potential offender tracking).  
- **Controls:**  
  - **Confidence Slider** – Fine-tune detection sensitivity.  
- **Output:**  
  - Detected bounding boxes.  
  - OCR text overlay for license plates or vest IDs.  

<p align="center">
  <img src="pictures_for_read_me/vest_violation_ex.jpg" alt="Vest Violation Example" height="300"/>
  <img src="pictures_for_read_me/no_vest_violation.jpg" alt="No Vest Violation Example" height="300"/>
</p>


---

### Dashboards  
- **No-Vest Detections**  
  - Tracks riders with missing vests in real time.  
  - Links detected riders to license plates.  
  - Displays simple compliance dashboard.  
- **Vest Violations**  
  - Detects individuals committing violations (no vest, altered vest).  
  - OCR extracts vest numbers for potential investigation.  
  - Dashboard summarizing detected non-compliant riders.  
- **Controls:**  
  - **Skip Frames, Resize Scale, Confidence, Update Rate** – Adjust real-time performance.  
- **Output:**  
  - Real-time detection overlay and violation statistics dashboard.  


##  Dashboard Examples  

<p align="center">
  <img src="pictures_for_read_me/dashboard_example.jpg" alt="Dashboard Example 1" width="45%"/>
  <img src="pictures_for_read_me/dashboard_example_three.jpg" alt="Dashboard Example 2" width="45%"/>
</p>

<p align="center">
  <img src="pictures_for_read_me/dashboard_vest_ex.jpg" alt="Vest Dashboard Example" width="45%"/>
  <img src="pictures_for_read_me/no_vest_ex.jpg" alt="No-Vest Example" width="45%"/>
</p>

---

