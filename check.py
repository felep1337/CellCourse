import numpy as np
import pickle
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path


def greedy_nn_displacements(c0, c1, max_dist=200):
    if not c0 or not c1:
        return np.array([]).reshape(0, 2)
    p0 = np.array([(c[0], c[1]) for c in c0])
    p1 = np.array([(c[0], c[1]) for c in c1])
    D = cdist(p0, p1)
    out = []
    for i in range(len(p0)):
        j = np.argmin(D[i])
        if D[i, j] < max_dist:
            out.append(p1[j] - p0[i])
    return np.array(out) if out else np.array([]).reshape(0, 2)


def main(pkl="diagnostics/detections.pkl", out="diagnostics/diag_clean.png"):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    centers = d["centers"]
    counts = [len(c) for c in centers]

    disps = []
    for t in range(len(centers) - 1):
        disps.append(greedy_nn_displacements(centers[t], centers[t + 1]))
    disps = np.vstack(disps) if any(len(d) for d in disps) else np.zeros((0, 2))
    speeds = np.linalg.norm(disps, axis=1) if len(disps) else np.array([])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].plot(counts, marker='.')
    ax[0].set_title(f"Детекций в кадре\nmedian={int(np.median(counts))}, "
                    f"CV={np.std(counts)/np.mean(counts):.2f}")
    ax[0].set_xlabel("кадр"); ax[0].grid(alpha=0.3)

    if len(speeds):
        ax[1].hist(speeds, bins=40)
        for p, c in [(50, 'g'), (90, 'orange'), (99, 'r')]:
            v = np.percentile(speeds, p)
            ax[1].axvline(v, color=c, ls='--', label=f'p{p}={v:.1f}')
        ax[1].set_xlabel("|Δr|, px/кадр"); ax[1].legend()
        ax[1].set_title("Перемещения (greedy NN)")

        ax[2].scatter(disps[:, 0], disps[:, 1], s=4, alpha=0.4)
        m = disps.mean(axis=0)
        ax[2].plot(m[0], m[1], 'r*', markersize=15, label=f'mean=({m[0]:+.1f}, {m[1]:+.1f})')
        ax[2].axhline(0, color='k', lw=0.5); ax[2].axvline(0, color='k', lw=0.5)
        ax[2].set_xlabel("dx"); ax[2].set_ylabel("dy"); ax[2].set_aspect('equal')
        ax[2].legend(); ax[2].set_title("Облако смещений")

    Path(out).parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"График: {out}")

    # Рекомендации
    if len(speeds):
        p99 = np.percentile(speeds, 99)
        median = np.median(speeds)
        drift = np.linalg.norm(disps.mean(axis=0))
        print(f"\nРекомендации для track.py:")
        print(f"  track_cost_cutoff       = {int((p99 * 1.2) ** 2)}    # = ({int(p99*1.2)} px)^2")
        print(f"  gap_closing_cost_cutoff = {int((p99 * 1.8) ** 2)}    # = ({int(p99*1.8)} px)^2")


if __name__ == "__main__":
    main()
