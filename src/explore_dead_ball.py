"""Exploración visual: ¿qué señales del tracking delatan un balón muerto?

Dibuja, para un tramo de frames, varias señales candidatas una debajo de otra:
  1. ¿se ve el balón?
  2. ¿está el balón fuera de la cancha?
  3. velocidad del balón (calculada desde su posición)
  4. velocidad media de los jugadores
  5. cuántos jugadores ve el tracking
  6. (opcional) la franja del ground truth PÚBLICO, solo para ENTENDER qué es "D"

El gráfico se guarda en data/cache/plots/ (ignorado por git). Es para mirarlo TÚ en tu PC:
no lo subas a ninguna IA ni al repo.

REGLA: esto es para entender y diseñar señales. Los umbrales NO se ajustan mirando el
ground truth público hasta que "cuadre" (ver RULES.md / EVALUATION.md).

Uso:
    python src/explore_dead_ball.py                       # los 5 min públicos (11185-18684)
    python src/explore_dead_ball.py --start 20000 --end 26000
    python src/explore_dead_ball.py --start 110808 --end 114500   # tramo antes del saque del 2T: TODO es D
    python src/explore_dead_ball.py --zoom      # un gráfico por cada tramo D del GT público (+-10 s)
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from atd_arrays import load_arrays  # noqa: E402

FPS = 25
PLOT_DIR = os.path.join(ROOT, "data", "cache", "plots")
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def ball_speed(ball, smooth_frames=5):
    """m/s a partir de la posición. NaN donde no hay balón en frames consecutivos."""
    d = np.full(len(ball), np.nan, dtype=np.float32)
    step = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    d[1:] = step
    if smooth_frames > 1:
        # media móvil ignorando NaN
        ok = np.isfinite(d)
        val = np.where(ok, d, 0.0)
        k = np.ones(smooth_frames)
        num = np.convolve(val, k, mode="same")
        den = np.convolve(ok.astype(float), k, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(den > 0, num / den, np.nan).astype(np.float32)
    return d


def features(A):
    L, W = A["pitch"]
    ball = A["ball"]
    visible = np.isfinite(ball[:, 0])
    with np.errstate(invalid="ignore"):
        out_of_pitch = visible & ((ball[:, 0] < 0) | (ball[:, 0] > L) |
                                  (ball[:, 1] < 0) | (ball[:, 1] > W))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # frames sin jugadores
        mean_player_speed = np.nanmean(A["speed"], axis=1)
    n_players = np.isfinite(A["players"][:, :, 0]).sum(axis=1)
    return {
        "visible": visible,
        "out_of_pitch": out_of_pitch,
        "ball_speed": ball_speed(ball),
        "mean_player_speed": mean_player_speed,
        "n_players": n_players,
    }


def load_public_gt():
    import pandas as pd
    gt = pd.read_csv(GT_PUBLIC, dtype={"label": "object"})["label"].str.strip().str.upper()
    return gt.to_numpy()


def plot(F, start, end, gt=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(start, end)
    secs = (t - start) / FPS
    rows = 6 if gt is not None else 5
    fig, ax = plt.subplots(rows, 1, figsize=(16, 2.0 * rows), sharex=True)

    ax[0].fill_between(secs, 0, F["visible"][start:end], step="mid", color="k")
    ax[0].set_ylabel("balón\nvisible")
    ax[1].fill_between(secs, 0, F["out_of_pitch"][start:end], step="mid", color="tab:orange")
    ax[1].set_ylabel("balón\nfuera")
    ax[2].plot(secs, F["ball_speed"][start:end], lw=0.7)
    ax[2].set_ylabel("vel. balón\n(m/s)")
    ax[2].set_ylim(0, 40)
    ax[3].plot(secs, F["mean_player_speed"][start:end], lw=0.7, color="tab:green")
    ax[3].set_ylabel("vel. media\njugadores")
    ax[4].plot(secs, F["n_players"][start:end], lw=0.7, color="tab:purple")
    ax[4].set_ylabel("nº jugadores\nvistos")

    if gt is not None:
        colours = {"H": "tab:red", "A": "tab:blue", "D": "0.6", "X": "white"}
        g = gt[start:end]
        for lab, col in colours.items():
            ax[5].fill_between(secs, 0, 1, where=(g == lab), step="mid", color=col, label=lab)
        ax[5].set_ylabel("GT público\n(solo mirar)")
        ax[5].legend(loc="upper right", ncol=4, fontsize=8)
        ax[5].set_yticks([])

    ax[-1].set_xlabel("segundos desde el frame {}".format(start))
    fig.suptitle("Señales candidatas de balón muerto · frames {}–{}".format(start, end))
    fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, "dead_ball_{}_{}.png".format(start, end))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def d_runs(gt, lo=11185, hi=18685):
    """Tramos [inicio, fin) donde el GT público es D (solo dentro de la ventana pública)."""
    is_d = np.zeros(len(gt) + 1, dtype=bool)
    is_d[lo:hi] = gt[lo:hi] == "D"
    edges = np.flatnonzero(np.diff(np.concatenate(([False], is_d))))
    return list(zip(edges[0::2], edges[1::2]))


def plot_zoom(F, gt, pad_s=10):
    """Un gráfico por tramo D: pad_s segundos antes y después, con el tramo sombreado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pad = pad_s * FPS
    paths = []
    for i, (a, b) in enumerate(d_runs(gt), 1):
        start, end = max(0, a - pad), min(len(gt), b + pad)
        secs = (np.arange(start, end) - a) / FPS   # 0 = empieza el D
        fig, ax = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
        series = [
            (F["visible"], "balón\nvisible", "k", True),
            (F["out_of_pitch"], "balón\nfuera", "tab:orange", True),
            (F["ball_speed"], "vel. balón\n(m/s)", "tab:blue", False),
            (F["mean_player_speed"], "vel. media\njugadores", "tab:green", False),
            (F["n_players"], "nº jugadores\nvistos", "tab:purple", False),
        ]
        for k, (arr, name, col, binary) in enumerate(series):
            y = arr[start:end]
            if binary:
                ax[k].fill_between(secs, 0, y, step="mid", color=col)
            else:
                ax[k].plot(secs, y, lw=0.9, color=col)
            ax[k].axvspan(0, (b - a) / FPS, color="0.85", zorder=0)
            ax[k].axvline(0, color="r", lw=1)
            ax[k].axvline((b - a) / FPS, color="g", lw=1)
            ax[k].set_ylabel(name)
        ax[2].set_ylim(0, 40)
        ax[-1].set_xlabel("segundos (0 = empieza D, línea roja; línea verde = se reanuda)")
        fig.suptitle("Tramo D #{} · frames {}–{} ({:.1f} s)".format(i, a, b, (b - a) / FPS))
        fig.tight_layout()
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, "zoom_D{:02d}.png".format(i))
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=11185)
    p.add_argument("--end", type=int, default=18685)
    p.add_argument("--no-gt", action="store_true", help="no dibujar la franja del GT público")
    p.add_argument("--zoom", action="store_true", help="un gráfico por tramo D del GT público")
    args = p.parse_args()

    A = load_arrays()
    F = features(A)
    if args.zoom:
        paths = plot_zoom(F, load_public_gt())
        print("{} gráficos guardados en data/cache/plots/zoom_DXX.png".format(len(paths)))
        print("míralos tú; no los subas a ninguna IA.")
        sys.exit(0)
    gt = None if args.no_gt else load_public_gt()
    path = plot(F, args.start, args.end, gt)
    print("gráfico guardado en:", os.path.relpath(path, ROOT))
    print("ábrelo en VS Code y míralo tú. No lo subas a ninguna IA.")
