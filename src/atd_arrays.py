"""Carga el ATD crudo (tactical) a arrays de numpy y lo cachea en data/cache/.

Por qué existe: kloppy tarda minutos y mucha RAM en leer 199k frames. Aquí se lee el .txt
una sola vez (como hace starter/baseline.py) y se guarda un .npz. Las siguientes veces carga
en segundos.

La caché vive dentro de data/, que está en el .gitignore -> nunca se sube al repo público.

Uso:
    from src.atd_arrays import load_arrays
    A = load_arrays()          # dict con ball, players, speed, team, ...

Índices: todo está indexado por FRAME DE VIDEO (0..199318) = fila de la submission.
(El ATD cuenta desde 1; aquí ya está corregido: video_frame = frame_id - 1.)
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "atd_tactical_arrays.npz")

HOME, AWAY = 0, 1


def _team_of_each_track(meta_path):
    """0 = home, 1 = away, por cada track slot (mismo criterio que starter/baseline.py)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("esperaba 2 equipos en {}, hay {}".format(meta_path, len(team_ids)))
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if tid == team_ids[0] else AWAY for tid in slots], dtype=np.int8)


def _parse_group(group):
    """'x,y,s' -> (x, y, s) en floats; NaN si falta. Tolera grupos con 2 o más valores."""
    vals = group.split(",")
    out = [np.nan, np.nan, np.nan]
    for i in range(min(3, len(vals))):
        try:
            out[i] = float(vals[i])
        except ValueError:
            out[i] = np.nan
    return out


def parse_raw(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team = _team_of_each_track(meta_path)
    n_slots = len(team)

    ball = np.full((n, 2), np.nan, dtype=np.float32)          # metros
    players = np.full((n, n_slots, 2), np.nan, dtype=np.float32)  # metros
    speed = np.full((n, n_slots), np.nan, dtype=np.float32)       # m/s (la que trae el ATD)

    print("leyendo {} (tarda 1-3 min la primera vez)...".format(os.path.basename(raw_path)))
    with open(raw_path) as fh:
        for k, line in enumerate(fh):
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1  # frame de video
            if not (0 <= f < n):
                continue
            groups = parts[1].split(";")
            for s, g in enumerate(groups[:n_slots]):
                x, y, sp = _parse_group(g)
                players[f, s, 0] = x * length
                players[f, s, 1] = y * width
                speed[f, s] = sp
            bx, by, _ = _parse_group(parts[2])
            ball[f, 0] = bx * length
            ball[f, 1] = by * width
            if k and k % 50000 == 0:
                print("  {} líneas".format(k))

    return {"ball": ball, "players": players, "speed": speed, "team": team,
            "pitch": np.array([length, width], dtype=np.float32)}


def load_arrays(game=GAME, data_dir=None, rebuild=False):
    if os.path.exists(CACHE_FILE) and not rebuild:
        with np.load(CACHE_FILE) as z:
            return {k: z[k] for k in z.files}
    arrays = parse_raw(game, data_dir)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(CACHE_FILE, **arrays)
    print("caché guardada en", os.path.relpath(CACHE_FILE, ROOT))
    return arrays


if __name__ == "__main__":
    A = load_arrays(rebuild="--rebuild" in sys.argv)
    n = len(A["ball"])
    seen = np.isfinite(A["ball"][:, 0]).sum()
    print("frames: {}   balón visible: {:.1f}%   slots: {}".format(
        n, 100.0 * seen / n, A["players"].shape[1]))
