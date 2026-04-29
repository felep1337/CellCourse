import numpy as np
import pickle
import czifile
import cv2
from cellpose import models, core
from scipy import ndimage as ndi
from pathlib import Path
import time
import matplotlib.pyplot as plt

CZI_PATH = "data_czi/12mmc+lncap.czi"
ROI_MASK_PNG = "diagnostics/roi_mask.png"
OUT_PKL = "diagnostics/detections.pkl"

MODEL_NAME = "nuclei"
DIAMETER = 18
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

AREA_MIN = 30
AREA_MAX = 800

MASK_INPUT_IMAGE = False

DEBUG_PREVIEW_FRAMES = 3
DEBUG_DIR = "calibration/segmentation_preview"


def load_czi_frames(filepath):
    with czifile.CziFile(filepath) as czi:
        arr = czi.asarray()
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        if arr.shape[-1] <= 4:
            arr = arr.mean(axis=-1)
        elif arr.shape[1] <= 4:
            arr = arr.mean(axis=1)
    return arr.astype(np.float32)


def normalize_for_cellpose(img):
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip((img - p1) / (p99 - p1 + 1e-10), 0, 1)
    return (img * 255).astype(np.uint8)


def apply_mask_to_image(img, mask):
    inside_median = int(np.median(img[mask > 0])) if (mask > 0).any() else int(img.mean())
    out = img.copy()
    out[mask == 0] = inside_median
    return out


def masks_to_centers(masks, area_min, area_max):
    centers = []
    if masks.max() == 0:
        return centers
    labels = np.arange(1, masks.max() + 1)
    coms = ndi.center_of_mass(masks > 0, masks, labels)
    areas = ndi.sum(masks > 0, masks, labels)
    for label, (cy, cx), area in zip(labels, coms, areas):
        if area_min <= area <= area_max:
            centers.append((float(cx), float(cy), float(area)))
    return centers


def filter_by_mask(centers, roi_mask):
    H, W = roi_mask.shape
    out = []
    for (x, y, area) in centers:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H and roi_mask[yi, xi] > 0:
            out.append((x, y, area))
    return out


def save_debug_preview(img, centers_raw, centers_filtered, roi_mask, out_path):
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.imshow(img, cmap='gray')

    H, W = roi_mask.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[roi_mask > 0] = [0, 1, 0, 0.15]
    ax.imshow(overlay)

    if centers_raw:
        xs = [c[0] for c in centers_raw]
        ys = [c[1] for c in centers_raw]
        ax.scatter(xs, ys, s=15, facecolors='none', edgecolors='gray',
                   linewidth=0.5, label=f'все ({len(centers_raw)})')
    if centers_filtered:
        xs = [c[0] for c in centers_filtered]
        ys = [c[1] for c in centers_filtered]
        ax.scatter(xs, ys, s=20, facecolors='none', edgecolors='red',
                   linewidth=1.2, label=f'прошли ROI ({len(centers_filtered)})')

    ax.set_title(f"{len(centers_raw)} детекций → {len(centers_filtered)} после ROI")
    ax.legend(loc='upper right')
    fig.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)


def main():
    Path(OUT_PKL).parent.mkdir(exist_ok=True, parents=True)
    Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(ROI_MASK_PNG).exists():
        print(f"ROI-маска не найдена: {ROI_MASK_PNG}")
        print("Сначала запусти 00_build_roi_mask.py")
        return

    roi_mask = cv2.imread(ROI_MASK_PNG, cv2.IMREAD_GRAYSCALE)
    print(f"ROI-маска загружена: {roi_mask.shape}, "
          f"покрытие {(roi_mask > 0).sum() / roi_mask.size * 100:.1f}%")

    print("\nЗагружаем кадры...")
    frames = load_czi_frames(CZI_PATH)
    print(f"Размерность: {frames.shape}")

    if frames.shape[1:] != roi_mask.shape:
        print(f"ОШИБКА: размер маски {roi_mask.shape} не совпадает с кадром {frames.shape[1:]}")
        return

    use_gpu = core.use_gpu()
    print(f"GPU: {use_gpu}")
    model = models.Cellpose(model_type=MODEL_NAME, gpu=use_gpu)

    print(f"\nСегментация ({MODEL_NAME}, diameter={DIAMETER})")
    print(f"MASK_INPUT_IMAGE = {MASK_INPUT_IMAGE}")

    all_centers = []
    t0 = time.time()
    n = frames.shape[0]
    for t in range(n):
        img_norm = normalize_for_cellpose(frames[t])
        img_for_seg = apply_mask_to_image(img_norm, roi_mask) if MASK_INPUT_IMAGE else img_norm

        masks, _, _, _ = model.eval(
            img_for_seg, diameter=DIAMETER, channels=[0, 0],
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD,
        )
        centers_raw = masks_to_centers(masks, AREA_MIN, AREA_MAX)
        centers = filter_by_mask(centers_raw, roi_mask)
        all_centers.append(centers)

        if t < DEBUG_PREVIEW_FRAMES:
            save_debug_preview(
                img_norm, centers_raw, centers, roi_mask,
                Path(DEBUG_DIR) / f"frame_{t:03d}.png"
            )

        elapsed = time.time() - t0
        eta = elapsed / (t + 1) * (n - t - 1)
        print(f"  {t+1}/{n}: {len(centers_raw)} → {len(centers)} клеток "
              f"(прошло {elapsed:.0f}с, осталось ~{eta:.0f}с)")

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"centers": all_centers, "frames_shape": frames.shape}, f)
    print(f"\nДетекции сохранены: {OUT_PKL}")
    print(f"Превью первых кадров: {DEBUG_DIR}/")
    print(f"Дальше: 02b_diagnose_clean.py → 03_track.py")


if __name__ == "__main__":
    main()
