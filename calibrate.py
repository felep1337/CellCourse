import numpy as np
import czifile
from cellpose import models, core
import matplotlib.pyplot as plt
from pathlib import Path
import time

CZI_PATH = "data_czi/12mmc+lncap.czi"
OUT_DIR = "calibration"

MODELS_TO_TRY = ["cyto3", "nuclei"]
DIAMETERS_TO_TRY = [12, 18, 25]
FRAMES_TO_CHECK = [0]


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


def overlay_masks(img, masks, ax, title):
    ax.imshow(img, cmap='gray')
    if masks.max() > 0:
        from skimage.segmentation import find_boundaries
        bnd = find_boundaries(masks, mode='outer')
        overlay = np.zeros((*img.shape, 4))
        overlay[bnd] = [1, 0, 0, 1]
        ax.imshow(overlay)
        n_cells = masks.max()
    else:
        n_cells = 0
    ax.set_title(f"{title}  →  {n_cells} клеток")
    ax.axis('off')


def calibrate():
    out = Path(OUT_DIR)
    out.mkdir(exist_ok=True)

    print("Загружаем CZI...")
    frames = load_czi_frames(CZI_PATH)
    n = frames.shape[0]
    print(f"Кадров: {n}, размер: {frames.shape[1:]}")

    global FRAMES_TO_CHECK
    if n >= 3 and len(FRAMES_TO_CHECK) == 1:
        FRAMES_TO_CHECK = [0, n // 2, n - 1]

    use_gpu = core.use_gpu()
    print(f"GPU: {use_gpu}")

    for model_name in MODELS_TO_TRY:
        print(f"\n=== Модель: {model_name} ===")
        try:
            t = time.time()
            model = models.Cellpose(model_type=model_name, gpu=use_gpu)
            print(f"  загружена за {time.time()-t:.1f}с")
        except Exception as e:
            print(f"Не удалось загрузить {model_name}: {e}")
            continue

        for frame_idx in FRAMES_TO_CHECK:
            img = normalize_for_cellpose(frames[frame_idx])

            fig, axes = plt.subplots(
                len(DIAMETERS_TO_TRY) + 1, 1,
                figsize=(20, 4 * (len(DIAMETERS_TO_TRY) + 1))
            )
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Сырой кадр {frame_idx}  ({img.shape[1]}×{img.shape[0]})")
            axes[0].axis('off')

            for i, d in enumerate(DIAMETERS_TO_TRY):
                print(f"  кадр {frame_idx}, diameter={d}...", end=" ", flush=True)
                t = time.time()
                try:
                    masks, flows, styles, diams = model.eval(
                        img,
                        diameter=d,
                        channels=[0, 0],
                        flow_threshold=0.4,
                        cellprob_threshold=0.0,
                    )
                    dt = time.time() - t
                    print(f"{masks.max()} клеток ({dt:.1f}с)")
                except Exception as e:
                    print(f"ошибка: {e}")
                    masks = np.zeros_like(img, dtype=np.int32)

                overlay_masks(img, masks, axes[i + 1], f"{model_name}, d={d}")

            plt.tight_layout()
            out_file = out / f"{model_name}_frame{frame_idx}.png"
            fig.savefig(out_file, dpi=80, bbox_inches='tight')
            plt.close(fig)
            print(f"  → сохранено: {out_file}")

    print(f"\nГотово. Открой PNG в {out}/, выбери лучшую (модель, diameter).")


if __name__ == "__main__":
    calibrate()
