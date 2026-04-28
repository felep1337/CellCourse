"""
Шаг 2: Трекинг через laptrack (Jaqaman LAP).

Алгоритм (де-факто стандарт для биологического частичного трекинга):
1. Frame-to-frame LAP: венгерский метод между парами (t, t+1) с матрицей
   стоимостей C[i,j] = ||p_i^t - p_j^{t+1}||^2 + штрафы.
   Включены "виртуальные" строки/столбцы для рождения/смерти трека —
   так алгоритм не обязан связывать всё подряд.

2. Gap closing LAP: второй проход на уровне фрагментов треков.
   Связывает обрывки, разделённые пропусками до max_gap кадров.
   Это и есть "минимизация суммарной траекторной энергии",
   близкая к глобальному оптимуму.

Без делений/слияний, как договорились.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from laptrack import LapTrack
import matplotlib.pyplot as plt


# ---- ПАРАМЕТРЫ (подставить после диагностики) ----
# max_distance:           99-й перцентиль смещений * 1.2
# gap_closing_max_distance: max_distance * 1.5..2
# gap_closing_max_frame_count: на сколько кадров клетка может "пропасть"
PARAMS = {
    "track_dist_metric": "sqeuclidean",   # квадратичная стоимость в духе lap-функционала
    "track_cost_cutoff": 2296 ,         # = max_distance^2 (т.к. sqeuclidean)
    "gap_closing_dist_metric": "sqeuclidean",
    "gap_closing_cost_cutoff": 5167,
    "gap_closing_max_frame_count": 3,
    "splitting_cost_cutoff": False,       # деления выключены
    "merging_cost_cutoff": False,         # слияния выключены
}


def detections_to_dataframe(all_centers):
    """Превращаем список списков центроидов в DataFrame для laptrack."""
    rows = []
    for t, centers in enumerate(all_centers):
        for (cx, cy, area) in centers:
            rows.append({"frame": t, "x": cx, "y": cy, "area": area})
    return pd.DataFrame(rows)


def run_tracking(detections_pkl="diagnostics/detections.pkl",
                 out_dir="results"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    with open(detections_pkl, "rb") as f:
        data = pickle.load(f)
    all_centers = data["centers"]
    frames_shape = data["frames_shape"]

    df = detections_to_dataframe(all_centers)
    print(f"Всего детекций: {len(df)} в {df['frame'].nunique()} кадрах")

    tracker = LapTrack(**PARAMS)
    track_df, split_df, merge_df = tracker.predict_dataframe(
        df,
        coordinate_cols=["x", "y"],
        frame_col="frame",
        only_coordinate_cols=False,  # сохраняем area и прочее
    )

    # track_df имеет MultiIndex: (frame, point_index_in_frame)
    # колонка "track_id" — итоговая траектория
    track_df = track_df.reset_index()
    print(f"\nТреков построено: {track_df['track_id'].nunique()}")

    # ---- Статистика ----
    stats = compute_track_stats(track_df)
    stats.to_csv(out / "track_stats.csv", index=False)
    print(f"Статистика: {out / 'track_stats.csv'}")
    print(stats.describe())

    # ---- Сохранение треков ----
    track_df.to_csv(out / "tracks.csv", index=False)
    print(f"Треки: {out / 'tracks.csv'}")

    # ---- Графики ----
    plot_tracks(track_df, frames_shape, out / "tracks_overview.png")
    plot_msd(track_df, out / "msd.png")

    return track_df, stats


def compute_track_stats(track_df):
    """Базовая статистика по каждой траектории."""
    rows = []
    for tid, sub in track_df.groupby("track_id"):
        sub = sub.sort_values("frame")
        if len(sub) < 2:
            # Одиночные детекции — track_length = 1, скоростей нет
            rows.append({
                "track_id": tid,
                "length": 1,
                "duration_frames": 1,
                "total_path": 0.0,
                "net_displacement": 0.0,
                "straightness": np.nan,
                "mean_speed": 0.0,
                "max_speed": 0.0,
            })
            continue
        xy = sub[["x", "y"]].to_numpy()
        steps = np.diff(xy, axis=0)
        step_lens = np.linalg.norm(steps, axis=1)
        total_path = step_lens.sum()
        net = np.linalg.norm(xy[-1] - xy[0])
        rows.append({
            "track_id": tid,
            "length": len(sub),
            "duration_frames": int(sub["frame"].max() - sub["frame"].min() + 1),
            "total_path": total_path,
            "net_displacement": net,
            "straightness": net / total_path if total_path > 0 else np.nan,
            "mean_speed": step_lens.mean(),
            "max_speed": step_lens.max(),
        })
    return pd.DataFrame(rows)


def plot_tracks(track_df, frames_shape, path, min_length=3):
    """Все траектории на одной картинке (отбрасываем шум длиной < min_length)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    H, W = frames_shape[1], frames_shape[2]
    ax.set_xlim(0, W); ax.set_ylim(H, 0)  # инверсия Y под изображение
    ax.set_aspect('equal')
    n_plotted = 0
    for tid, sub in track_df.groupby("track_id"):
        if len(sub) < min_length:
            continue
        sub = sub.sort_values("frame")
        ax.plot(sub["x"], sub["y"], lw=0.8, alpha=0.7)
        n_plotted += 1
    ax.set_title(f"Траектории (длина ≥ {min_length}): {n_plotted}")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Карта траекторий: {path}")


def plot_msd(track_df, path, min_length=10, max_lag=20):
    """Mean Squared Displacement — диагностика типа движения.
    Линейный MSD(τ) → диффузия. MSD ~ τ^2 → направленное движение."""
    fig, ax = plt.subplots(figsize=(8, 6))
    msd_curves = []
    for tid, sub in track_df.groupby("track_id"):
        if len(sub) < min_length:
            continue
        sub = sub.sort_values("frame")
        xy = sub[["x", "y"]].to_numpy()
        n = len(xy)
        msd = []
        for lag in range(1, min(max_lag, n)):
            d = xy[lag:] - xy[:-lag]
            msd.append((d ** 2).sum(axis=1).mean())
        msd_curves.append(msd)

    if msd_curves:
        max_len = max(len(m) for m in msd_curves)
        # Усредняем по трекам (выравниваем NaN'ами по длине)
        padded = np.full((len(msd_curves), max_len), np.nan)
        for i, m in enumerate(msd_curves):
            padded[i, :len(m)] = m
        mean_msd = np.nanmean(padded, axis=0)
        lags = np.arange(1, max_len + 1)

        ax.loglog(lags, mean_msd, 'o-', label='ensemble MSD')
        # Подгонка степенной зависимости MSD ~ τ^α
        valid = ~np.isnan(mean_msd) & (mean_msd > 0)
        if valid.sum() > 3:
            log_t = np.log(lags[valid])
            log_msd = np.log(mean_msd[valid])
            alpha, _ = np.polyfit(log_t, log_msd, 1)
            ax.set_title(f"MSD, показатель α = {alpha:.2f}\n"
                         f"(α≈1: диффузия, α≈2: направленное движение)")
        ax.set_xlabel("lag, кадров")
        ax.set_ylabel("MSD, px²")
        ax.grid(alpha=0.3, which='both')
        ax.legend()

    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"MSD: {path}")


if __name__ == "__main__":
    run_tracking()
