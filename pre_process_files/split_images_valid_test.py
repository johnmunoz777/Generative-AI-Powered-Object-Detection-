import cv2, random
from pathlib import Path

def split_and_resize_for_yolo(
    src=r"C:\Users\john\yolo_peru_project\valid_real_pictures",
    train_out=r"C:\Users\john\yolo_peru_project\peru_train",
    test_out=r"C:\Users\johnm\yolo_peru_project\peru_test",
    n_each=220,
    max_long_side=1920,   
    seed=42
):
    """
    Randomly selects 440 images from the source folder (using the given seed),
    splits them into 220 for training and 220 for testing, resizes each while
    preserving aspect ratio (max side = 1920), and saves them into the train_out
    and test_out folders with their original filenames.
    """
    src = Path(src); train_out = Path(train_out); test_out = Path(test_out)
    train_out.mkdir(parents=True, exist_ok=True); test_out.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if len(imgs) < 2 * n_each:
        raise ValueError(f"Need at least {2*n_each} images; found {len(imgs)}")

    rng = random.Random(seed)
    chosen = rng.sample(imgs, 2 * n_each)
    rng.shuffle(chosen)
    train_files = chosen[:n_each]
    test_files  = chosen[n_each:]

    def resize_keep_aspect(img, max_side):
        if max_side is None: return img
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side <= max_side: return img
        scale = max_side / float(long_side)
        return cv2.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_AREA)

    def save_resized(src_path: Path, dst_dir: Path):
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"[skip] unreadable: {src_path.name}")
            return 0
        img = resize_keep_aspect(img, max_long_side)
        out_path = dst_dir / src_path.name  # keep original filename
        # set jpeg quality for jpg/jpeg; png/webp will ignore this param safely
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return 1

    t1 = sum(save_resized(p, train_out) for p in train_files)
    t2 = sum(save_resized(p, test_out)  for p in test_files)

    print(f"[DONE] Train: {t1} images → {train_out}")
    print(f"[DONE] Test : {t2} images → {test_out}")

# Example run:
split_and_resize_for_yolo()
