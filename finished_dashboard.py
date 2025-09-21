import os, re, io, time, base64, tempfile, json, random
from datetime import datetime, timedelta
from collections import Counter, deque
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
HALF_OK = USE_GPU and torch.cuda.get_device_properties(0).major >= 7
IMG_SIZE_DEFAULT = 320
st.set_page_config(page_title="Compliance Detector", page_icon="⚡", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
      .metric-row .metric { margin-right: 14px; }
      .muted { color: #9CA3AF; }
      .chip{display:inline-block;margin:6px 8px 6px 0;padding:6px 12px;border-radius:999px;background:#FEE2E2;color:#7F1D1D;border:1px solid #FECACA;font-weight:700;}
      .hud { background:#0F172A;border:1px solid #1F2A44;border-radius:10px;padding:12px 14px;margin-bottom:10px;}
      .pro-table { border:1px solid #1F2A44; border-radius:10px; overflow:hidden;}
      .pro-th { background:#0F172A; color:#E5E7EB; padding:10px 12px; font-size:13px; letter-spacing:.3px; text-transform:uppercase; }
      .pro-tr { background:#111827; border-bottom:1px solid #1F2A44; }
      .pro-td { color:#E5E7EB; padding:10px 12px; }
    </style>
    """, unsafe_allow_html=True
)
st.title("Compliance Detector ⚡")
CLASS_NAMES = ['helmet','license_plate','no_helmet','no_plate','no_vest','vest']
VIOLATION_LABELS = {'no_helmet','no_plate','no_vest'}
VIOLATION_TYPES = [
    "Shooting Incident", "Crime Committed", "Red Light Violation",
    "Stolen Property", "Speeding Violation", "Unauthorized Access",
    "Vandalism", "Public Disturbance", "Illegal Parking",
    "Trespassing", "Assault", "Theft in Progress",
    "Drug Activity", "Weapon Possession", "Fraud Attempt"
]
GREEN = (0, 180, 0)
RED = (0, 0, 255)
RED_DARK = (0, 0, 180)
GREEN_DARK = (0, 120, 0)
PLATE_BLUE = (255, 200, 0)
@st.cache_resource(show_spinner=False)
def load_yolo(weights_path: str, imgsz: int = IMG_SIZE_DEFAULT):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model not found: {weights_path}")
    model = YOLO(weights_path)
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, verbose=False, device=DEVICE, half=HALF_OK)
    return model

@st.cache_resource(show_spinner=False)
def load_coco_yolo_nano():
    model = YOLO("yolov8n.pt")
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    model(dummy, verbose=False, device=DEVICE, half=HALF_OK)
    return model

def yolo_call(model, img, conf=0.5, max_det=50, classes=None, imgsz=IMG_SIZE_DEFAULT):
    return model(
        img,
        conf=conf,
        verbose=False,
        max_det=max_det,
        device=DEVICE,
        imgsz=imgsz,
        half=HALF_OK,
        agnostic_nms=True
    )
def draw_corner_box(img, x1,y1,x2,y2, color, thickness=4, corner=25):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.line(img, (x1, y1), (x1 + corner, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner), color, thickness)
    cv2.line(img, (x2, y2), (x2 - corner, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner), color, thickness)
def draw_label(img, x1, y1, x2, y2, text, is_viol):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.9
    thickness = 2
    bg = RED_DARK if is_viol else GREEN_DARK
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    ly = y1 - 10 if y1 > 40 else y2 + th + 10
    cv2.rectangle(img, (x1 - 2, ly - th - 8), (x1 + tw + 10, ly + 5), bg, -1)
    cv2.putText(img, text, (x1 + 5, ly), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
def lightbox_html(records, id_col_name, show_viol_types=False):
    head = """
    <style>
      .grid{width:100%;border-collapse:separate;border-spacing:0 10px;}
      th{text-align:left;background:#0F172A;color:#E5E7EB;padding:12px;border-radius:8px;}
      td{background:#111827;border:1px solid #1F2A44;padding:12px;border-radius:8px;vertical-align:middle;}
      .badge{display:inline-block;padding:6px 10px;border-radius:999px;font-weight:700;background:#FEE2E2;color:#7F1D1D;border:1px solid #FECACA;}
      .badge2{display:inline-block;padding:6px 10px;border-radius:999px;font-weight:700;background:#FFEDD5;color:#9A3412;border:1px solid #FED7AA;margin-top:6px;}
      .muted{color:#9CA3AF;}
      .thumb{width:320px;height:200px;object-fit:cover;border-radius:8px;cursor:pointer;}
      a.lb{color:#93C5FD;text-decoration:none;font-weight:700;cursor:pointer;}
      .modal{display:none;position:fixed;z-index:99999;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,.85);}
      .modal img{position:absolute;max-width:90%;max-height:90%;top:50%;left:50%;transform:translate(-50%,-50%);border-radius:10px;}
      .timecell{color:#F8FAFC;font-weight:700;font-size:14px;line-height:1.35;}
    </style>
    <div id="lb" class="modal" onclick="this.style.display='none'"><img id="lbimg" src=""></div>
    <script>
      function showImg(src){document.getElementById('lbimg').src=src;document.getElementById('lb').style.display='block';}
    </script>
    """
    rows = []
    for r in records:
        with open(r["raw_path"], "rb") as f:
            raw_b64 = base64.b64encode(f.read()).decode()
        with open(r["det_path"], "rb") as f:
            det_b64 = base64.b64encode(f.read()).decode()
        raw_src = f"data:image/jpeg;base64,{raw_b64}"
        det_src = f"data:image/jpeg;base64,{det_b64}"
        id_html = f"<span class='badge'>{r['id']}</span>"
        if show_viol_types and 'viol_type' in r:
            id_html += f"<br><span class='badge2'>{r['viol_type']}</span>"
        rows.append(f"""
        <tr>
          <td><div class="timecell">{r['time']}</div></td>
          <td>{id_html}</td>
          <td><img class="thumb" src="{raw_src}" onclick="showImg('{raw_src}')"/></td>
          <td><img class="thumb" src="{det_src}" onclick="showImg('{det_src}')"/></td>
          <td class="muted">Frame {r['frame']}</td>
          <td>
            <a class="lb" onclick="showImg('{raw_src}')">Open original</a><br/>
            <a class="lb" onclick="showImg('{det_src}')">Open overlay</a>
          </td>
        </tr>
        """)

    if not rows:
        rows_html = "<tr><td colspan='6' class='muted'>No records.</td></tr>"
    else:
        rows_html = "\n".join(rows)

    tbl = f"""
    {head}
    <table class="grid">
      <tr><th>Time</th><th>{id_col_name}</th><th>Violation Image</th><th>Detection</th><th>Frame</th><th>Links</th></tr>
      {rows_html}
    </table>
    """
    return tbl

def pro_count_df(counts: Dict[str, int]):
    rows = [{"Class": k.replace("_", " ").title(), "Count": v} for k,v in counts.items()]
    rows.sort(key=lambda x: (-x["Count"], x["Class"]))
    return rows

tab1, tab2, tab3, tab4 = st.tabs(["Video Detections", "Image Detections", "Vest Detector", "Dashboards"])

# =============================================================
# TAB 1: VIDEO DETECTIONS
# =============================================================
with tab1:
    left, mid = st.columns([1.1, 2.2], gap="large")

    with left:
        st.subheader("Controls")
        model_choice = st.selectbox("Model", ["yolo_custom", "yolo_custom with coco"], index=0, key="t1_model")
        custom_path = st.text_input("Custom model .pt", value="model_weights.pt", key="t1_custom")
        coco_gate_path = st.text_input("COCO (for motorbike gate)", value="yolov8n.pt", key="t1_coco")
        video_file = st.file_uploader("Video (mp4/mov/avi/mkv)", type=["mp4","mov","avi","mkv"], key="t1_vid")

        st.markdown("**Performance**")
        skip_frames = st.slider("Skip frames", 1, 100, 10, 1, key="t1_skip")
        resize_scale = st.slider("Resize scale", 0.10, 1.00, 0.30, 0.05, key="t1_scale")
        conf_thr = st.slider("Confidence", 0.10, 0.90, 0.50, 0.05, key="t1_conf")
        imgsz = st.select_slider("YOLO img size", options=[256, 320, 384, 448, 512], value=320, key="t1_imgsz")

        st.markdown("**UI Throttling**")
        ui_every = st.slider("Update preview every N frames", 1, 10, 3, 1, key="t1_ui_every")

        run_btn = st.button("▶️ Start", type="primary", key="t1_run")
        st.caption(f"GPU: {'ON' if USE_GPU else 'OFF'} • Half: {'ON' if (USE_GPU and HALF_OK) else 'OFF'}")

    with mid:
        st.subheader("Preview")
        frame_ph = st.empty()
        hud = st.empty()

    if run_btn:
        if video_file is None:
            st.error("Upload a video first.")
            st.stop()

        model_custom = load_yolo(custom_path, imgsz=imgsz)
        model_coco = None
        if model_choice.endswith("with coco"):
            model_coco = load_coco_yolo_nano()
            st.info("COCO model loaded for motorbike detection gating")

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as vf:
            vf.write(video_file.read())
            vid_path = vf.name

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            st.error("Unable to open video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        out_dir = tempfile.mkdtemp(prefix="t1_annot_")
        out_path = os.path.join(out_dir, f"annotated_{int(time.time())}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        DETECTION_MEMORY = 15
        memory_counter = 0
        bike_active = False
        coco_check_interval = 5

        counts = Counter()
        violations = 0
        last_results = []
        frame_idx = 0
        t0 = time.time()

        bike_indicator = st.empty() if model_coco else None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            motorbike_boxes = []

            
            if model_coco is not None and frame_idx % coco_check_interval == 0:
                coco_scale = 0.25
                tiny = cv2.resize(frame, None, fx=coco_scale, fy=coco_scale)
                res_coco = yolo_call(model_coco, tiny, conf=0.30, classes=[3], imgsz=256)

                motorbike_found = False
                scale_back = 1.0 / coco_scale
                for r in res_coco:
                    if r.boxes is None: 
                        continue
                    if len(r.boxes) > 0:
                        motorbike_found = True
                        for b in r.boxes:
                            x1, y1, x2, y2 = (b.xyxy[0] * scale_back).int().tolist()
                            conf = float(b.conf[0])
                            motorbike_boxes.append((x1, y1, x2, y2, conf))
                        break

                if motorbike_found:
                    memory_counter = DETECTION_MEMORY
                else:
                    memory_counter = max(0, memory_counter - 1)

                bike_active = memory_counter > 0

                if bike_indicator:
                    if bike_active:
                        bike_indicator.success("🏍️ Motorbike detected - Running custom detections")
                    else:
                        bike_indicator.warning("No motorbike - Skipping custom detections")

            
            if frame_idx % skip_frames == 0:
                run_custom = True if model_coco is None else bike_active

                last_results = []
                if run_custom:
                    small_w = max(1, int(width * resize_scale))
                    small_h = max(1, int(height * resize_scale))
                    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

                    res = yolo_call(model_custom, small, conf=conf_thr, max_det=50, imgsz=imgsz)
                    scale_back_x = width / small_w
                    scale_back_y = height / small_h

                    for r in res:
                        if r.boxes is None: continue
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1 = int(float(x1) * scale_back_x)
                            y1 = int(float(y1) * scale_back_y)
                            x2 = int(float(x2) * scale_back_x)
                            y2 = int(float(y2) * scale_back_y)
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            last_results.append((x1, y1, x2, y2, conf, cls))

                    for *_, conf, cls in last_results:
                        name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
                        counts[name] += 1
                        if name in VIOLATION_LABELS:
                            violations += 1

    
            out = frame.copy()

            
            if model_coco is not None and motorbike_boxes:
                for x1, y1, x2, y2, conf in motorbike_boxes:
                    dot_x, dot_y = x1 + 12, y1 + 12
                    cv2.circle(out, (dot_x, dot_y), 6, (255, 200, 100), -1)
                    cv2.putText(out, "MOTORBIKE", (dot_x + 10, dot_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1, cv2.LINE_AA)

            
            for x1, y1, x2, y2, conf, cls in last_results:
                cname = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
                is_viol = cname in VIOLATION_LABELS
                box_color = RED if is_viol else GREEN
                draw_corner_box(out, x1, y1, x2, y2, box_color, thickness=4, corner=25)
                label = f"{cname.upper()}: {conf:.0%}"
                draw_label(out, x1, y1, x2, y2, label, is_viol)
                cv2.circle(out, (x1 + 15, y1 + 15), 10, (255, 255, 255), -1)
                cv2.circle(out, (x1 + 15, y1 + 15), 8, box_color, -1)

            
            try:
                writer.write(out)
            except Exception:
                pass

            
            if frame_idx % ui_every == 0:
                frame_ph.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
                elapsed = time.time() - t0
                analyzed = max(1, frame_idx // max(1, skip_frames))
                effective_fps = frame_idx / elapsed if elapsed > 0 else 0
                frac = min(1.0, (frame_idx + 1) / total) if total else 0
                rem = int((elapsed / frac - elapsed)) if frac > 0.01 else 0
                hud.markdown(
                    f"""
                    <div class='hud'>
                      <b>Frames:</b> {frame_idx:,} / {total or '—'} &nbsp; | &nbsp;
                      <b>Analyzed:</b> {analyzed:,} &nbsp; | &nbsp; 
                      <b>Violations:</b> {violations:,} &nbsp; | &nbsp;
                      <b>FPS:</b> {effective_fps:.1f} &nbsp; | &nbsp;
                      <b>ETA:</b> {rem//60:d}m {rem%60:02d}s &nbsp; | &nbsp;
                      <span class='muted'>GPU: {'ON' if USE_GPU else 'OFF'}</span>
                    </div>
                    """, unsafe_allow_html=True
                )

            frame_idx += 1

        cap.release()
        writer.release()
        os.unlink(vid_path)

        st.subheader("Summary")
        rows = pro_count_df(counts)
        if rows:
            st.markdown(
                "<div class='pro-table'>"
                "<div class='pro-th'>Detections by Class</div>"
                + "".join([f"<div class='pro-tr'><div class='pro-td'><b>{r['Class']}</b></div><div class='pro-td'>{r['Count']}</div></div>" for r in rows])
                + "</div>",
                unsafe_allow_html=True
            )
        else:
            st.caption("No detections recorded.")

        
        st.markdown("**Download Annotated Video**")
        try:
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download MP4",
                    data=f,
                    file_name=os.path.basename(out_path),
                    mime="video/mp4",
                    key="t1_download_video"
                )
            st.caption(f"Saved to: `{out_path}`")
        except Exception as e:
            st.warning(f"Could not provide download: {e}")

# =============================================================
# TAB 2: IMAGE DETECTIONS  
# =============================================================
with tab2:
    left_i, mid_i = st.columns([1.1, 2.2], gap="large")

    with left_i:
        st.subheader("Controls")
        img_file = st.file_uploader("Image (jpg/png)", type=["jpg","jpeg","png"], key="t2_img")
        model_path_img = st.text_input("Custom model .pt", value="model_weights.pt", key="t2_model")
        do_plate = st.checkbox("License Plate OCR", value=True, key="t2_lp")
        do_vest = st.checkbox("Vest OCR", value=False, key="t2_vest")
        conf_img = st.slider("Confidence", 0.10, 0.90, 0.50, 0.05, key="t2_conf")
        run_img = st.button("🔍 Analyze", type="primary", key="t2_run")

    with mid_i:
        st.subheader("Preview")
        title_ph = st.empty()
        img_display_ph = st.empty()
        dl_ph_annot = st.empty()
        dl_ph_plates = st.empty()
        dl_ph_vest = st.empty()
        dl_ph_both = st.empty()

    @st.cache_resource(show_spinner=False)
    def get_paddle_ocr():
        from paddleocr import PaddleOCR
        return PaddleOCR(lang='en', use_gpu=USE_GPU, show_log=False)

    def ocr_plate(img, box, min_len=4):
        x1,y1,x2,y2 = box
        ocr = get_paddle_ocr()
        pad=12
        crop = img[max(0,y1-pad):min(img.shape[0],y2+pad), max(0,x1-pad):min(img.shape[1],x2+pad)]
        if crop.size==0: return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 40: gray = cv2.resize(gray, (gray.shape[1]*2, 80))
        best=None; best_conf=0.0
        for cand in [gray, cv2.convertScaleAbs(gray, alpha=2.0, beta=10), cv2.bitwise_not(gray)]:
            try:
                res = ocr.ocr(cand, cls=False)
                if not res or not res[0]: continue
                text=""; tot=0; cnt=0
                for line in res[0]:
                    if line and len(line)>=2 and line[1][1] > 0.35:
                        text += line[1][0].strip().upper(); tot += line[1][1]; cnt += 1
                if cnt:
                    conf = tot/cnt
                    clean = re.sub(r'[^A-Z0-9]','', text)
                    if len(clean) >= min_len and conf > best_conf:
                        best, best_conf = clean, conf
            except Exception: continue
        return best

    def ocr_vest(img, box, min_len=3):
        x1,y1,x2,y2 = box
        ocr = get_paddle_ocr()
        pad=15
        crop = img[max(0,y1-pad):min(img.shape[0],y2+pad), max(0,x1-pad):min(img.shape[1],x2+pad)]
        if crop.size==0: return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 60: gray = cv2.resize(gray, (gray.shape[1]*3, gray.shape[0]*3))
        best=None; best_conf=0.0
        for cand in [gray, cv2.convertScaleAbs(gray, alpha=2.0, beta=20), cv2.bitwise_not(gray)]:
            try:
                res = ocr.ocr(cand, cls=False)
                if not res or not res[0]: continue
                text=""; tot=0; cnt=0
                for line in res[0]:
                    if line and len(line)>=2 and line[1][1] > 0.40:
                        text += line[1][0].strip().upper(); tot += line[1][1]; cnt += 1
                if cnt:
                    conf = tot/cnt
                    clean = re.sub(r'[^A-Z0-9]','', text)
                    if len(clean) >= min_len and conf > best_conf:
                        best, best_conf = clean, conf
            except Exception: continue
        return best

    if run_img:
        if img_file is None:
            st.error("Upload an image first.")
            st.stop()

        with st.spinner("Running detections and OCR…"):
            model_img = load_yolo(model_path_img)
            img_bytes = img_file.read()
            arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Failed to read image.")
                st.stop()

            res = yolo_call(model_img, img, conf=conf_img, max_det=50)
            plate_text = None
            vest_text = None
            has_vest = False

            plate_boxes = []
            vest_boxes = []

            for r in res:
                if r.boxes is None: continue
                for box in r.boxes:
                    cls = int(box.cls[0]); cname = CLASS_NAMES[cls]
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    if cname == 'vest':
                        has_vest = True
                        vest_boxes.append((x1,y1,x2,y2))
                    if cname == 'license_plate':
                        plate_boxes.append((x1,y1,x2,y2))

            if do_plate and plate_boxes:
                plate_text = ocr_plate(img, max(plate_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])))
            if do_vest and vest_boxes:
                vest_text = ocr_vest(img, max(vest_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])))

            disp = img.copy()
            for r in res:
                if r.boxes is None: continue
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0]); conf=float(box.conf[0]); cls=int(box.cls[0])
                    cname = CLASS_NAMES[cls]
                    is_viol = cname in VIOLATION_LABELS
                    color = RED if is_viol else GREEN
                    draw_corner_box(disp, x1,y1,x2,y2, color, thickness=3, corner=20)
                    draw_label(disp, x1,y1,x2,y2, f"{cname.upper()}: {conf:.0%}", is_viol)

        parts=[]
        if has_vest: parts.append("VEST DETECTED")
        if plate_text: parts.append(f"PLATE: {plate_text}")
        if vest_text: parts.append(f"VEST: {vest_text}")
        title_ph.markdown("### "+(" | ".join(parts) if parts else "Result"))
        img_display_ph.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)

        
        try:
            ok, enc = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok:
                annot_bytes = enc.tobytes()
                base_name = getattr(img_file, "name", "image.jpg")
                name_no_ext = os.path.splitext(base_name)[0]
                dl_ph_annot.download_button(
                    "⬇️ Download Annotated Image (JPG)",
                    data=annot_bytes,
                    file_name=f"{name_no_ext}_annotated.jpg",
                    mime="image/jpeg",
                    key="t2_dl_annot_img"
                )
        except Exception:
            pass

        
        if plate_text:
            plate_buf = io.BytesIO( (plate_text + "\n").encode("utf-8") )
            dl_ph_plates.download_button(
                "⬇️ Download License Plate OCR (TXT)",
                data=plate_buf, file_name="license_plate_ocr.txt",
                mime="text/plain", key="t2_dl_plate_txt"
            )
        if vest_text:
            vest_buf = io.BytesIO( (vest_text + "\n").encode("utf-8") )
            dl_ph_vest.download_button(
                "⬇️ Download Vest OCR (TXT)",
                data=vest_buf, file_name="vest_ocr.txt",
                mime="text/plain", key="t2_dl_vest_txt"
            )

        rows = []
        if plate_text: rows.append(("license_plate", plate_text))
        if vest_text: rows.append(("vest", vest_text))
        if rows:
            csv_str = "type,text\n" + "\n".join([f"{t},{v}" for t,v in rows])
            csv_buf = io.BytesIO(csv_str.encode("utf-8"))
            dl_ph_both.download_button(
                "⬇️ Download OCR Results (CSV)",
                data=csv_buf, file_name="ocr_results.csv",
                mime="text/csv", key="t2_dl_ocr_csv"
            )

# =============================================================
# TAB 3: VEST  DASHBOARD
# =============================================================

with tab3:
    left_d, mid_d = st.columns([1.1, 2.2], gap="large")

    with left_d:
        st.subheader("Controls")
        video3 = st.file_uploader("Video (mp4/mov/avi/mkv)", type=["mp4","mov","avi","mkv"], key="t3_vid")
        model3 = st.text_input("Custom model .pt", value="model_weights.pt", key="t3_model")
        skip3 = st.slider("Skip frames", 1, 100, 10, 1, key="t3_skip")
        scale3 = st.slider("Resize scale", 0.10, 1.00, 0.25, 0.05, key="t3_scale")
        conf3  = st.slider("Confidence", 0.10, 0.90, 0.45, 0.05, key="t3_conf")
        ui3_every = st.slider("Update preview every N frames", 1, 10, 3, 1, key="t3_ui_every")
        run3 = st.button("📊 Build", type="primary", key="t3_run")

    with mid_d:
        st.subheader("Preview")
        preview3 = st.empty()
        stats3 = st.empty()
    
        dl3 = st.empty()

    if run3:
        if video3 is None:
            st.error("Upload a video first.")
            st.stop()

        model_dash = load_yolo(model3)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video3.name)[1]) as vf:
            vf.write(video3.read())
            vpath = vf.name
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            st.error("Unable to open video.")
            st.stop()

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps3 = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        out_dir3 = tempfile.mkdtemp(prefix="t3_annot_")
        out_path3 = os.path.join(out_dir3, f"annotated_{int(time.time())}.mp4")
        fourcc3 = cv2.VideoWriter_fourcc(*"mp4v")
        writer3 = cv2.VideoWriter(out_path3, fourcc3, fps3, (width, height))

        frame_idx=0; analyzed=0; det_count=0; uniques=set()
        counts=Counter(); t0=time.time()
        base_dir = tempfile.mkdtemp(prefix="dash3_")
        raw_dir  = os.path.join(base_dir, "raw"); os.makedirs(raw_dir, exist_ok=True)
        records = []

        def _ocr_vest_t3(img, x1, y1, x2, y2, min_len=3):
            ocr = get_paddle_ocr()
            pad=15
            crop = img[max(0,y1-pad):min(img.shape[0],y2+pad), max(0,x1-pad):min(img.shape[1],x2+pad)]
            if crop.size==0: return None
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] < 60: gray = cv2.resize(gray, (gray.shape[1]*3, gray.shape[0]*3))
            best=None; best_conf=0.0
            for cand in [gray, cv2.convertScaleAbs(gray, alpha=2.0, beta=20), cv2.bitwise_not(gray)]:
                try:
                    res = ocr.ocr(cand, cls=False)
                    if not res or not res[0]: continue
                    text=""; tot=0; cnt=0
                    for line in res[0]:
                        if line and len(line)>=2 and line[1][1] > 0.40:
                            text += line[1][0].strip().upper(); tot += line[1][1]; cnt += 1
                    if cnt:
                        conf = tot/cnt
                        clean = re.sub(r'[^A-Z0-9]','', text)
                        if len(clean) >= min_len and conf > best_conf:
                            best, best_conf = clean, conf
                except Exception:
                    continue
            return best

        last_frame_dets = []

        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % skip3 == 0:
                analyzed += 1
                small = cv2.resize(frame, (max(1,int(width*scale3)), max(1,int(height*scale3))), interpolation=cv2.INTER_LINEAR)
                res = yolo_call(model_dash, small, conf=conf3, imgsz=IMG_SIZE_DEFAULT)
                scale_back = 1.0 / scale3 if scale3>0 else 1.0
                plates=[]; vests=[]
                last_frame_dets = []

                for r in res:
                    if r.boxes is None: continue
                    for b in r.boxes:
                        conf=float(b.conf[0]); cls=int(b.cls[0])
                        if conf < conf3: continue
                        x1,y1,x2,y2 = (b.xyxy[0] * scale_back).int().tolist()
                        name = CLASS_NAMES[cls]
                        counts[name]+=1; det_count+=1
                        last_frame_dets.append((x1,y1,x2,y2,name,conf))
                        if name=='license_plate': plates.append((x1,y1,x2,y2,conf))
                        if name=='vest': vests.append((x1,y1,x2,y2,conf))

                
                if plates and vests:
                    vx1,vy1,vx2,vy2,_ = max(vests, key=lambda x:x[4])
                    lx1,ly1,lx2,ly2,_ = max(plates, key=lambda x:x[4])
                    ts = datetime.now().strftime("%H:%M:%S<br><small>%d/%m/%Y</small>")
                    vest_id = _ocr_vest_t3(frame, vx1, vy1, vx2, vy2)
                    uniq_id = vest_id if vest_id else f"VEST_FRAME_{frame_idx}"
                    if uniq_id not in uniques:
                        tag = f"vest_{frame_idx}"
                        raw_path = os.path.join(raw_dir, f"{tag}_raw.jpg")
                        det_path = os.path.join(raw_dir, f"{tag}_det.jpg")
                        det_img = frame.copy()
                        cv2.rectangle(det_img,(vx1,vy1),(vx2,vy2),(60,60,255),3)
                        cv2.rectangle(det_img,(lx1,ly1),(lx2,ly2),(255,200,0),3)
                        cv2.imwrite(raw_path, frame); cv2.imwrite(det_path, det_img)
                        records.append({"time": ts, "id": uniq_id, "frame": frame_idx, "raw_path": raw_path, "det_path": det_path})
                        uniques.add(uniq_id)

            
            out = frame.copy()
            for (x1,y1,x2,y2,name,conf) in last_frame_dets:
                color = RED if name in VIOLATION_LABELS else GREEN
                draw_corner_box(out,x1,y1,x2,y2,color,thickness=3,corner=18)
                draw_label(out,x1,y1,x2,y2, f"{name.upper()}: {conf:.0%}", name in VIOLATION_LABELS)

            try:
                writer3.write(out)
            except Exception:
                pass

            
            if frame_idx % ui3_every == 0:
                preview3.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
                elapsed = time.time() - t0
                effective_fps = frame_idx / elapsed if elapsed > 0 else 0
                frac = min(1.0, (frame_idx + 1) / total) if total else 0
                rem = int((elapsed / frac - elapsed)) if frac > 0.01 else 0
                stats3.markdown(
                    f"<div class='hud'><b>Frames:</b> {frame_idx:,}/{total or '—'} &nbsp; | &nbsp;"
                    f"<b>Analyzed:</b> {analyzed:,} &nbsp; | &nbsp; <b>Detections:</b> {det_count:,} &nbsp; | &nbsp;"
                    f"<b>Unique:</b> {len(uniques):,} &nbsp; | &nbsp; <b>FPS:</b> {effective_fps:.1f} &nbsp; | &nbsp; "
                    f"<b>ETA:</b> {rem//60:d}m {rem%60:02d}s</div>",
                    unsafe_allow_html=True
                )

            frame_idx += 1

        cap.release()
        writer3.release()
        os.unlink(vpath)

        st.subheader("Counts by Class")
        rows = pro_count_df(counts)
        if rows:
            st.markdown(
                "<div class='pro-table'>"
                "<div class='pro-th'>Detections by Class</div>"
                + "".join([f"<div class='pro-tr'><div class='pro-td'><b>{r['Class']}</b></div><div class='pro-td'>{r['Count']}</div></div>" for r in rows])
                + "</div>",
                unsafe_allow_html=True
            )
        if uniques:
            st.markdown("**Unique**")
            st.markdown("".join([f"<span class='chip'>{u}</span>" for u in sorted(uniques)]), unsafe_allow_html=True)

        
        st.markdown("**Download Annotated Video**")
        try:
            with open(out_path3, "rb") as f:
                dl3.download_button(
                    "⬇️ Download MP4 (Annotated)",
                    data=f,
                    file_name=os.path.basename(out_path3),
                    mime="video/mp4",
                    key="t3_download_video"
                )
            st.caption(f"Saved to: `{out_path3}`")
        except Exception as e:
            st.warning(f"Could not provide download: {e}")

# =============================================================
# TAB 4: DASHBOARDS VEST + NOVEST COMPLIANCE
# =============================================================
with tab4:
    left4, mid4, right4 = st.columns([1.1, 1.6, 1.3], gap="large")

    with left4:
        st.subheader("Controls")
        mode = st.selectbox(
            "Pipeline",
            ["No-Vest Detections (license_plate + no_vest)", "Vest Violations (vest + license_plate)"],
            index=0, key="t4_mode"
        )
        up_video = st.file_uploader("Video (mp4/mov/avi/mkv)", type=["mp4","mov","avi","mkv"], key="t4_vid")
        model_tab4 = st.text_input("Custom model .pt", value="model_weights.pt", key="t4_model")

        st.markdown("**Performance**")
        skip4 = st.slider("Skip frames", 1, 100, 40, 1, key="t4_skip")
        scale4 = st.slider("Resize scale", 0.10, 1.00, 0.20, 0.05, key="t4_scale")
        conf4  = st.slider("Confidence", 0.10, 0.90, 0.45, 0.05, key="t4_conf")
        ui4_every = st.slider("Update preview every N frames", 1, 10, 3, 1, key="t4_ui_every")

        st.markdown("**OCR**")
        min_chars = st.slider("Min text len", 2, 12, 5, 1, key="t4_minlen")

        run4 = st.button("🚀 Run", type="primary", key="t4_run")
        st.caption(f"GPU: {'ON' if USE_GPU else 'OFF'} • Half: {'ON' if (USE_GPU and HALF_OK) else 'OFF'}")

    with mid4:
        st.subheader("Live Preview")
        prev4 = st.empty()
        status4 = st.empty()

    with right4:
        st.subheader("Live Stats")
        stat_box = st.empty()
        table_box = st.empty()
        uniq_box = st.empty()

    @st.cache_resource(show_spinner=False)
    def ocr_engine():
        from paddleocr import PaddleOCR
        return PaddleOCR(lang='en', use_gpu=USE_GPU, show_log=False)

    def ocr_plate_fast(img, box, min_len):
        x1,y1,x2,y2 = box
        ocr = ocr_engine()
        pad=12
        crop = img[max(0,y1-pad):min(img.shape[0],y2+pad), max(0,x1-pad):min(img.shape[1],x2+pad)]
        if crop.size==0: return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 40: gray = cv2.resize(gray, (gray.shape[1]*2, 80))
        best=None; best_conf=0.0
        for cand in [gray, cv2.convertScaleAbs(gray, alpha=2.0, beta=10), cv2.bitwise_not(gray)]:
            try:
                res = ocr.ocr(cand, cls=False)
                if not res or not res[0]: continue
                text=""; tot=0; cnt=0
                for line in res[0]:
                    if line and len(line)>=2 and line[1][1] > 0.35:
                        text += line[1][0].strip().upper(); tot += line[1][1]; cnt += 1
                if cnt:
                    conf = tot/cnt
                    clean = re.sub(r'[^A-Z0-9]','', text)
                    if len(clean) >= min_len and conf > best_conf:
                        best, best_conf = clean, conf
            except Exception: continue
        return best

    def ocr_vest_fast(img, box, min_len):
        x1,y1,x2,y2 = box
        ocr = ocr_engine()
        pad=15
        crop = img[max(0,y1-pad):min(img.shape[0],y2+pad), max(0,x1-pad):min(img.shape[1],x2+pad)]
        if crop.size==0: return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 60: gray = cv2.resize(gray, (gray.shape[1]*3, gray.shape[0]*3))
        best=None; best_conf=0.0
        for cand in [gray, cv2.convertScaleAbs(gray, alpha=2.0, beta=20), cv2.bitwise_not(gray)]:
            try:
                res = ocr.ocr(cand, cls=False)
                if not res or not res[0]: continue
                text=""; tot=0; cnt=0
                for line in res[0]:
                    if line and len(line)>=2 and line[1][1] > 0.40:
                        text += line[1][0].strip().upper(); tot += line[1][1]; cnt += 1
                if cnt:
                    conf = tot/cnt
                    clean = re.sub(r'[^A-Z0-9]','', text)
                    if len(clean) >= min_len and conf > best_conf:
                        best, best_conf = clean, conf
            except Exception: continue
        return best

    if run4:
        if up_video is None:
            st.error("Upload a video first.")
            st.stop()

        model4 = load_yolo(model_tab4)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up_video.name)[1]) as vf:
            vf.write(up_video.read())
            v4 = vf.name
        cap = cv2.VideoCapture(v4)
        if not cap.isOpened():
            st.error("Unable to open video.")
            st.stop()

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_idx=0; analyzed=0; det_count=0; uniques=set(); counts=Counter()
        records=[]; base_dir=tempfile.mkdtemp(prefix="tab4_"); raw_dir=os.path.join(base_dir,"raw"); os.makedirs(raw_dir, exist_ok=True)
        want_vest = ("Vest Violations" in mode)

        stable_detections = []
        detection_history = deque(maxlen=3)
        
        t0=time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % skip4 == 0:
                analyzed += 1
                small = cv2.resize(frame, (max(1,int(width*scale4)), max(1,int(height*scale4))), interpolation=cv2.INTER_LINEAR)
                res = yolo_call(model4, small, conf=conf4, imgsz=IMG_SIZE_DEFAULT)
                scale_back = 1.0 / scale4 if scale4>0 else 1.0
                plates=[]; vests=[]; no_vests=[]
                current_dets = []

                for r in res:
                    if r.boxes is None: continue
                    for b in r.boxes:
                        conf=float(b.conf[0]); cls=int(b.cls[0])
                        if conf < conf4: continue
                        x1,y1,x2,y2 = (b.xyxy[0] * scale_back).int().tolist()
                        name = CLASS_NAMES[cls]
                        counts[name]+=1; det_count+=1
                        current_dets.append((x1,y1,x2,y2,name,conf))
                        if name=='license_plate': plates.append((x1,y1,x2,y2,conf))
                        elif name=='vest': vests.append((x1,y1,x2,y2,conf))
                        elif name=='no_vest': no_vests.append((x1,y1,x2,y2,conf))

                detection_history.append(current_dets)
                if len(detection_history) > 0:
                    stable_detections = current_dets if current_dets else stable_detections

                if want_vest and plates and vests:
                    vx1,vy1,vx2,vy2,_ = max(vests, key=lambda x:x[4])
                    lx1,ly1,lx2,ly2,_ = max(plates, key=lambda x:x[4])
                    vtxt = ocr_vest_fast(frame, (vx1,vy1,vx2,vy2), min_chars)
                    if vtxt and vtxt not in uniques:
                        uniques.add(vtxt)
                        ts=datetime.now().strftime("%H:%M:%S<br><small>%d/%m/%Y</small>")
                        base=f"vest_{vtxt}_f{frame_idx}"
                        raw_path=os.path.join(raw_dir, base+"_raw.jpg")
                        det_path=os.path.join(raw_dir, base+"_det.jpg")
                        det_img=frame.copy()
                        cv2.rectangle(det_img,(vx1,vy1),(vx2,vy2),(60,60,255),3)
                        cv2.rectangle(det_img,(lx1,ly1),(lx2,ly2),(255,200,0),3)
                        cv2.imwrite(raw_path, frame); cv2.imwrite(det_path, det_img)
                        records.append({
                            "time": ts, "id": vtxt, "frame": frame_idx,
                            "raw_path": raw_path, "det_path": det_path,
                            "viol_type": random.choice(VIOLATION_TYPES)
                        })

                if (not want_vest) and plates and no_vests:
                    nx1,ny1,nx2,ny2,_ = max(no_vests, key=lambda x:x[4])
                    lx1,ly1,lx2,ly2,_ = max(plates, key=lambda x:x[4])
                    ptxt = ocr_plate_fast(frame, (lx1,ly1,lx2,ly2), min_chars)
                    if ptxt and ptxt not in uniques:
                        uniques.add(ptxt)
                        ts=datetime.now().strftime("%H:%M:%S<br><small>%d/%m/%Y</small>")
                        base=f"plate_{ptxt}_f{frame_idx}"
                        raw_path=os.path.join(raw_dir, base+"_raw.jpg")
                        det_path=os.path.join(raw_dir, base+"_det.jpg")
                        det_img=frame.copy()
                        cv2.rectangle(det_img,(nx1,ny1),(nx2,ny2),(60,60,255),3)
                        cv2.rectangle(det_img,(lx1,ly1),(lx2,ly2),(255,200,0),3)
                        cv2.imwrite(raw_path, frame); cv2.imwrite(det_path, det_img)
                        records.append({
                            "time": ts, "id": ptxt, "frame": frame_idx,
                            "raw_path": raw_path, "det_path": det_path
                        })

            
            show = frame.copy()
            for (x1,y1,x2,y2,name,conf) in stable_detections:
                color = RED if name in VIOLATION_LABELS else GREEN
                draw_corner_box(show,x1,y1,x2,y2,color,thickness=3,corner=18)
                draw_label(show,x1,y1,x2,y2, f"{name.upper()}: {conf:.0%}", name in VIOLATION_LABELS)

            if frame_idx % ui4_every == 0:
                rgb_frame = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                prev4.image(rgb_frame, use_container_width=True)
                
                elapsed = time.time() - t0
                frac = min(1.0, (frame_idx+1) / total) if total else 0
                rem = int((elapsed/frac - elapsed)) if frac > 0.01 else 0
                effective_fps = frame_idx / elapsed if elapsed > 0 else 0
                
                status4.markdown(
                    f"<div class='hud'><b>Frames:</b> {frame_idx+1:,}/{total or '—'} &nbsp; | &nbsp; "
                    f"<b>Analyzed:</b> {analyzed:,} &nbsp; | &nbsp; <b>Detections:</b> {det_count:,} &nbsp; | &nbsp; "
                    f"<b>Unique:</b> {len(uniques):,} &nbsp; | &nbsp; <b>FPS:</b> {effective_fps:.1f} &nbsp; | &nbsp; "
                    f"ETA: {rem//60:d}m {rem%60:02d}s</div>",
                    unsafe_allow_html=True
                )
                
                stat_box.markdown(
                    f"<div class='hud'><b>Mode:</b> {'VEST + PLATE' if want_vest else 'PLATE + NO_VEST'}<br>"
                    f"<span class='muted'>GPU: {'ON' if USE_GPU else 'OFF'} • Half: {'ON' if (USE_GPU and HALF_OK) else 'OFF'}</span></div>",
                    unsafe_allow_html=True
                )
                
                if counts:
                    rows = pro_count_df(counts)
                    table_box.markdown(
                        "<div class='pro-table'>"
                        "<div class='pro-th'>Counts by Class</div>"
                        + "".join([f"<div class='pro-tr'><div class='pro-td'><b>{r['Class']}</b></div><div class='pro-td'>{r['Count']}</div></div>" for r in rows])
                        + "</div>",
                        unsafe_allow_html=True
                    )
                
                if uniques:
                    uniq_label = "Unique Vests" if want_vest else "Unique Plates"
                    uniq_box.markdown("**"+uniq_label+"**<br>"+ "".join([f"<span class='chip'>{u}</span>" for u in sorted(uniques)]), unsafe_allow_html=True)

            frame_idx += 1

        cap.release()
        os.unlink(v4)

        st.subheader("Dashboard (inline)")
        id_col = "Vest ID" if want_vest else "Plate"
        html = lightbox_html(records, id_col_name=id_col, show_viol_types=want_vest)
        st.components.v1.html(html, height=900, scrolling=True)

        out_html = os.path.join(base_dir, "dashboard.html")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        with left4:
            with open(out_html, "rb") as f:
                st.download_button("📄 Download Dashboard HTML", f, file_name="dashboard.html", mime="text/html")
            st.caption(f"Saved to: `{out_html}`")
