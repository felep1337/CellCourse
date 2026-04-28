"""
Построение ROI-маски вручную: обводим область ВНЕ трубки,
где реально есть клетки. Маска сохраняется в PNG (бинарная).

Использование:
    1. Запусти этот скрипт.
    2. Откроется окно с кадром.
    3. Левый клик — добавить точку полигона.
       Правый клик — отменить последнюю точку.
       Двойной левый клик — закрыть полигон (соединит с первой точкой).
    4. После закрытия первого полигона можно начать второй (если зон несколько).
    5. Закрой окно когда закончил — маска сохранится.

ВАЖНО: обводи именно те зоны, ГДЕ ХОЧЕШЬ детектить клетки
(то есть вне трубки). Всё остальное будет отфильтровано.
"""
import numpy as np
import czifile
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon as MplPolygon
import cv2
from pathlib import Path

CZI_PATH = "data_czi/12mmc+lncap.czi"
FRAME_FOR_MASK = 1   # на каком кадре строить (трубка одна и та же на всех)
OUT_MASK_PNG = "diagnostics/roi_mask.png"
OUT_PREVIEW_PNG = "diagnostics/roi_mask_preview.png"


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


def normalize(img):
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip((img - p1) / (p99 - p1 + 1e-10), 0, 1)
    return (img * 255).astype(np.uint8)


def main():
    Path(OUT_MASK_PNG).parent.mkdir(exist_ok=True, parents=True)

    print("Загружаем CZI...")
    frames = load_czi_frames(CZI_PATH)
    img = normalize(frames[FRAME_FOR_MASK])
    H, W = img.shape
    print(f"Размер: {W}x{H}")

    # Список полигонов (можно добавить несколько)
    polygons = []

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(img, cmap='gray')
    ax.set_title(
        f"Кадр {FRAME_FOR_MASK}: обведи область ГДЕ ЕСТЬ КЛЕТКИ (вне трубки).\n"
        "ЛКМ — точка, ПКМ — отменить, Esc — сбросить, Enter — закрыть полигон."
    )

    drawn_patches = []

    def on_select(verts):
        """Вызывается когда пользователь закрыл полигон."""
        if len(verts) < 3:
            return
        polygons.append(verts)
        # Рисуем закрашенный полигон полупрозрачно
        patch = MplPolygon(verts, closed=True, alpha=0.3,
                           facecolor='lime', edgecolor='lime', linewidth=2)
        ax.add_patch(patch)
        drawn_patches.append(patch)
        fig.canvas.draw_idle()
        print(f"Полигон #{len(polygons)} добавлен ({len(verts)} точек). "
              f"Можно начать новый или закрыть окно.")
        # Перезапускаем селектор для следующего полигона
        nonlocal selector
        selector.disconnect_events()
        selector = PolygonSelector(ax, on_select, useblit=True,
                                   props=dict(color='yellow', linewidth=2))

    selector = PolygonSelector(ax, on_select, useblit=True,
                               props=dict(color='yellow', linewidth=2))

    print("\nКликай точки полигона. Двойной клик / Enter — закрыть полигон.")
    print("Можно построить несколько полигонов подряд.")
    print("Когда всё готово — просто закрой окно.\n")
    plt.show()

    if not polygons:
        print("Нет полигонов, маска не сохранена.")
        return

    # Строим бинарную маску: 1 внутри полигонов (где есть клетки), 0 снаружи
    mask = np.zeros((H, W), dtype=np.uint8)
    for verts in polygons:
        pts = np.array(verts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    cv2.imwrite(OUT_MASK_PNG, mask)
    print(f"\nМаска сохранена: {OUT_MASK_PNG}")
    print(f"Полигонов: {len(polygons)}, площадь маски: "
          f"{(mask > 0).sum() / mask.size * 100:.1f}% от кадра")

    # Превью: исходник + маска полупрозрачно
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(img, cmap='gray')
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[mask > 0] = [0, 1, 0, 0.3]
    ax.imshow(overlay)
    ax.set_title(f"ROI-маска: зелёное — где будем искать клетки")
    fig.savefig(OUT_PREVIEW_PNG, dpi=80, bbox_inches='tight')
    plt.close(fig)
    print(f"Превью: {OUT_PREVIEW_PNG}")


if __name__ == "__main__":
    main()
