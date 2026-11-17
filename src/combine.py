"""Integración. Responsable: Jeremy.

Junta balón muerto (D) + posesión (H/A) en una etiqueta por frame:
  - fuera de las dos mitades calificadas -> X (no se califica)
  - dentro, D donde dead_ball dice D; si no, H/A de possession
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from possession import HOME, AWAY, assign  # noqa: E402


def in_play_window(n, game=GAME):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def labels(S, dead, game=GAME):
    n = S["n"]
    window = in_play_window(n, game)
    alive = window & ~dead
    team = assign(S, alive)
    out = np.full(n, "X", dtype="<U1")
    out[window & dead] = "D"
    out[alive & (team == HOME)] = "H"
    out[alive & (team == AWAY)] = "A"
    return out


def write_csv(path, lab):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(lab))
    return path
