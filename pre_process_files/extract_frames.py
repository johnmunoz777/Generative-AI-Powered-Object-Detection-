import cv2
import random
from pathlib import Path

def extract_random_frames(
    src_dir="my_videos",
    out_dir="individual_frames",
    frames_per_video=15,
    seed=42,
    resize_to=(1280, 720)  # set to None to keep original size
):
    """
    Extracts N random frames from each video in src_dir and writes them into a single out_dir.
    Filenames are sequential across all videos: image_one.jpg, image_two.jpg, ...
    
    Args:
        src_dir (str or Path): Folder containing your videos (e.g., VEo 3 outputs).
        out_dir (str or Path): Folder where all extracted frames will be stored together.
        frames_per_video (int): How many random frames to take from each video.
        seed (int): Random seed for reproducibility.
        resize_to (tuple or None): (width, height). Use None to keep original size.
                                  
    Returns:
        int: total number of frames written.
    """

    
    def number_to_words(n: int) -> str:
        ones = ["zero","one","two","three","four","five","six","seven","eight","nine"]
        teens = ["ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
        tens = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

        if n < 10:
            return ones[n]
        if 10 <= n < 20:
            return teens[n-10]
        if 20 <= n < 100:
            t, o = divmod(n, 10)
            return tens[t] if o == 0 else f"{tens[t]}_{ones[o]}"
        if 100 <= n < 1000:
            h, r = divmod(n, 100)
            return f"{ones[h]}_hundred" if r == 0 else f"{ones[h]}_hundred_{number_to_words(r)}"
        if 1000 <= n < 10000:
            th, r = divmod(n, 1000)
            return f"{ones[th]}_thousand" if r == 0 else f"{ones[th]}_thousand_{number_to_words(r)}"
        return str(n)

    
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
    rng = random.Random(seed)

    videos = [p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTS]

    print(f"[info] Found {len(videos)} video(s) in {src_dir}")

    img_index = 1
    total_written = 0

    # ----- process each video -----
    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"[skip] cannot open: {vid.name}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if frame_count <= 0:
            print(f"[skip] zero frames: {vid.name}")
            cap.release()
            continue

        k = min(frames_per_video, frame_count)
        indices = sorted(rng.sample(range(frame_count), k))  # random unique frames

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                ok2, frame = cap.read()
                if not ok2 or frame is None:
                    print(f"[warn] failed read @ {vid.name} frame {idx}")
                    continue

            if resize_to is not None:
                w, h = resize_to
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            word = number_to_words(img_index)
            out_path = out_dir / f"image_{word}.jpg"
            if cv2.imwrite(str(out_path), frame):
                total_written += 1
                img_index += 1
            else:
                print(f"[warn] failed write: {out_path.name}")

        cap.release()

    final_count = len(list(out_dir.glob("*.jpg")))
    print(f"[done] wrote {total_written} new images")
    print(f"[info] total images now in {out_dir}: {final_count}")

    return total_written



extract_random_frames(
     src_dir="my_videos",
     out_dir="individual_frames",
     frames_per_video=15,
     seed=42,
     resize_to=(1280, 720)  
 )