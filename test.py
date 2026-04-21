import numpy as np
import czifile
import napari
from skimage import filters
import cv2
from scipy import ndimage as ndi


def segment_cells(image):
    image = np.power(image, 0.7)

    img = (image - image.min()) / (image.max() - image.min() + 1e-10)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    img = np.clip(img - cv2.GaussianBlur(img, (51, 51), 0), 0, 1)

    img = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1)

    binary = img > filters.threshold_otsu(img)

    binary = ndi.binary_fill_holes(binary)

    contours, _ = cv2.findContours((binary * 255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        if 20 < area < 500:
            x, y, w, h = cv2.boundingRect(c)
            if max(w, h) / (min(w, h) + 1e-5) < 3:
                filtered.append(c)

    return filtered


class KalmanTrack:
    def __init__(self, x, y):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([[1,0,1,0],
                                             [0,1,0,1],
                                             [0,0,1,0],
                                             [0,0,0,1]], np.float32)

        self.kf.measurementMatrix = np.array([[1,0,0,0],
                                              [0,1,0,0]], np.float32)

        self.kf.statePre = np.array([[x],[y],[0],[0]], np.float32)

        self.points = [(int(x), int(y))]

def process_multiple_frames_czi(filepath):
    print(f"Файл {filepath}")

    with czifile.CziFile(filepath) as czi:
        image_data = czi.asarray()
        print(f"Размерность: {image_data.shape}")

        n_frames = image_data.shape[0]
        viewer = napari.Viewer(title="Трекинг клеток")

        masks = []
        images = []

        for frame in range(1):
            img = np.mean(image_data[frame, :, :, :], axis=2).astype(np.float32)

            contours = segment_cells(img)

            centers = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cx = x + w // 2
                cy = y + h // 2
                centers.append([cx, cy])

            mask = np.zeros_like(img, dtype=np.uint8)
            for contour in contours:
                cv2.polylines(mask, [contour.astype(np.int32)], True, 255, 2)

            masks.append(mask)
            images.append(img)

        traj = np.zeros_like(images[0], dtype=np.float32)


        viewer.add_image(np.stack(images), name="Оригинал")
        viewer.add_image(np.stack(masks), name="Контуры",colormap='red', blending='additive')

        napari.run()

if __name__ == "__main__":
    czi_file_path = "data_czi/12mmc+lncap.czi"
    process_multiple_frames_czi(czi_file_path)