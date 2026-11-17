"""Módulo de balón muerto (D). Responsable: Miguel.

Devuelve un array booleano `dead` por frame de video. Reglas (todas automáticas):

  R1 Saque inicial: dentro de cada mitad, todo lo anterior al saque inicial es D. El saque se
     ubica con el tag Smart Tagging "Kick Off (Start)" y se afina al primer frame de esa
     ventana en que el balón arranca a moverse.
  R2 Balón fuera: desde que el balón cruza una línea (más de OUT_MARGIN m) hasta la siguiente
     reanudación. La reanudación es el siguiente balón parado del Smart Tagging (si llega en
     <= MAX_DEAD_S), o si no, el primer momento en que el balón vuelve a la cancha y se mueve.
  R3 Balón parado sin salida (faltas): antes de cada reanudación del Smart Tagging que no venga
     ya precedida de D, se marca D desde que la actividad de los jugadores cayó.
  R4 Parón largo: balón invisible >= LONG_MISSING_S y jugadores casi quietos -> D.
  R5 Limpieza: se rellenan huecos vivos < 1 s entre D y se borran D < 1 s.

Nada de esto usa el ground truth: los umbrales son conocimiento de fútbol fijado a priori.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, require  # noqa: E402
from signals import FPS, runs  # noqa: E402

# Params (fijados a priori, ver TECHNICAL_DOC.md)
MIN_OUT_FRAMES = 5          # el balón debe estar fuera >= 0.2 s seguidos
MAX_DEAD_S = 90             # un balón muerto no dura más de 90 s
FALLBACK_DEAD_S = 60        # tope si no se encuentra la reanudación
KICK_SPEED = 3.0            # m/s: el balón "arranca" (saque) por encima de esto
KICK_FRAMES = 3             # ...durante al menos 3 frames
ACT_HIGH = 1.6              # m/s: actividad media de jugadores típica de juego vivo
ACT_LOW = 1.2               # m/s: jugadores casi quietos
STOP_LOOKBACK_S = 45        # R3: cuánto mirar hacia atrás desde la reanudación
MIN_STOP_S = 2.0            # R3: duración mínima de un D antes de una reanudación
LONG_MISSING_S = 3.0        # R4
MIN_D_S = 1.0               # R5
MIN_ALIVE_GAP_S = 1.0       # R5


def load_restarts(game=GAME, data_dir=None):
    """Balones parados del Smart Tagging: [{start, end, type, half, team}] en frames de video."""
    import xml.etree.ElementTree as ET
    path = require(["smart_tagging"], game, data_dir)[0]
    out = []
    for inst in ET.parse(path).iter("instance"):
        code = (inst.findtext("code") or "").strip().lower()
        if code != "set pieces":
            continue
        tags = {lab.findtext("group"): (lab.findtext("text") or "")
                for lab in inst.findall("label")}
        out.append({
            "start": int(round(float(inst.findtext("start")) * FPS)),
            "end": int(round(float(inst.findtext("end")) * FPS)),
            "type": tags.get("Type", "").strip().lower(),
            "half": tags.get("Half", "").strip(),
            "team": tags.get("Team", "").strip(),
        })
    return sorted(out, key=lambda r: r["start"])


def kick_frame(S, a, b):
    """Primer frame en [a, b) donde el balón se mueve > KICK_SPEED durante KICK_FRAMES. None si no."""
    a, b = max(0, a), min(S["n"], b)
    moving = np.nan_to_num(S["ball_speed"][a:b], nan=0.0) > KICK_SPEED
    run = 0
    for i, m in enumerate(moving):
        run = run + 1 if m else 0
        if run >= KICK_FRAMES:
            return a + i - KICK_FRAMES + 1
    return None


def restart_frame(S, r):
    """Frame de la reanudación dentro de la ventana del tag (~5 s); si no, el centro del tag."""
    pad = FPS  # 1 s de margen a cada lado
    k = kick_frame(S, r["start"] - pad, r["end"] + pad)
    return k if k is not None else (r["start"] + r["end"]) // 2


def detect(S, restarts, game=GAME):
    n = S["n"]
    dead = np.zeros(n, dtype=bool)
    blocks = half_video_blocks(game)
    restart_frames = sorted(restart_frame(S, r) for r in restarts)

    # R1: antes del saque inicial de cada mitad
    for half, first, last in blocks:
        name = "1st Half" if half == 1 else "2nd Half"
        kos = [r for r in restarts if r["type"] == "kick off (start)" and r["half"] == name]
        if not kos:
            kos = [r for r in restarts if r["type"] == "kick off (start)" and first <= r["start"] <= last]
        if kos:
            ko = restart_frame(S, kos[0])
            if first < ko <= last:
                dead[first:ko] = True

    # R2: balón fuera -> hasta la reanudación
    rf = np.array(restart_frames, dtype=np.int64)
    back_in = S["inside"] & (np.nan_to_num(S["ball_speed"], nan=0.0) > KICK_SPEED)
    for a, b in runs(S["out"]):
        if b - a < MIN_OUT_FRAMES:
            continue
        nxt = rf[(rf > a) & (rf <= a + MAX_DEAD_S * FPS)]
        if len(nxt):
            end = int(nxt[0])
        else:
            cand = np.flatnonzero(back_in[b:min(n, a + FALLBACK_DEAD_S * FPS)])
            end = b + int(cand[0]) if len(cand) else min(n, a + FALLBACK_DEAD_S * FPS)
        dead[a:end] = True

    # R3: reanudaciones sin D previo (faltas, etc.)
    act = np.nan_to_num(S["activity"], nan=0.0)
    for r in restart_frames:
        if r <= 0 or dead[r - 1]:
            continue
        lo = max(0, r - STOP_LOOKBACK_S * FPS)
        high = np.flatnonzero(act[lo:r] >= ACT_HIGH)
        start = lo + int(high[-1]) + 1 if len(high) else lo
        start = min(start, r - int(MIN_STOP_S * FPS))
        dead[max(0, start):r] = True

    # R4: parón largo con balón invisible y jugadores quietos
    for a, b in runs(~S["visible"]):
        if b - a >= LONG_MISSING_S * FPS and np.nanmean(S["activity"][a:b]) < ACT_LOW:
            dead[a:b] = True

    # R5: limpieza
    for a, b in runs(~dead):
        if 0 < a and b < n and (b - a) < MIN_ALIVE_GAP_S * FPS:
            dead[a:b] = True
    for a, b in runs(dead):
        if (b - a) < MIN_D_S * FPS:
            dead[a:b] = False

    return dead, restart_frames
