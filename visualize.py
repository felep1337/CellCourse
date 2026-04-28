import numpy as np
import pandas as pd
import czifile
import napari

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


def visualize(czi_path, tracks_csv="results/tracks.csv", min_length=20):
    frames = load_czi_frames(czi_path)
    df = pd.read_csv(tracks_csv)

    keep = df.groupby("track_id").filter(lambda g: len(g) >= min_length)
    tracks_np = keep[["track_id", "frame", "y", "x"]].to_numpy()

    viewer = napari.Viewer(title="Cell tracking")
    viewer.add_image(frames, name="raw", contrast_limits=[frames.min(), frames.max()])
    viewer.add_tracks(tracks_np, name=f"tracks (≥{min_length} кадров)",
                       tail_length=20, head_length=0)

    points = df[["frame", "y", "x"]].to_numpy()
    viewer.add_points(points, name="detections", size=8,
                      face_color="transparent")

    napari.run()

if __name__ == "__main__":
    visualize("data_czi/12mmc+lncap.czi")
