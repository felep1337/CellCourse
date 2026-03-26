import numpy as np
import czifile
import napari
from skimage import filters
from skimage.segmentation import clear_border
import cv2
from scipy import ndimage as ndi
from skimage import measure

def gamma_correction(image):
    enhanced = np.power(image, 2)
    return enhanced

def segment_cells(image):
    gamma_correction(image)

    img_normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)

    img_blurred = cv2.GaussianBlur(img_normalized.astype(np.float32), (5, 5), 1)

    threshold_value = filters.threshold_otsu(img_blurred)

    binary = img_blurred > threshold_value

    binary = ndi.binary_fill_holes(binary)
    labeled = measure.label(binary)
    labeled = clear_border(labeled)
    binary = labeled > 0

    binary_uint8 = (binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def process_multiple_frames_czi(filepath):
    print(f"Файл {filepath}")
    with czifile.CziFile(filepath) as czi:
        image_data = czi.asarray()
        print(f"Размерность: {image_data.shape}")

        n_frames = image_data.shape[0]
        viewer = napari.Viewer(title="Клеточки:3")

        masks = []
        images = []

        for frame in range(n_frames):
            img = np.mean(image_data[frame, :, :, :], axis=2).astype(np.float32)
            contours = segment_cells(img)
            mask = np.zeros_like(img, dtype=np.uint8)
            for contour in contours:
                cv2.polylines(mask, [contour.astype(np.int32)], True, 255, thickness=2)
            masks.append(mask)
            images.append(img)

        viewer.add_image(np.stack(images), name="Оригинал", colormap="gray")
        viewer.add_image(np.stack(masks), name="Контур", colormap='blue',
                         blending='additive', opacity=0.7)

        napari.run()

if __name__ == "__main__":
    czi_file_path = "data_czi/12mmc+lncap.czi"
    process_multiple_frames_czi(czi_file_path)