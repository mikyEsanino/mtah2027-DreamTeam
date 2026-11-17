"""Módulo de posesión H/A. Responsable: Kheyla.

Idea: la posesión solo cambia cuando el rival logra un control REAL del balón, no con un toque,
un rebote o un duelo. Pasos, dentro de cada tramo de juego vivo:

  1. Candidato por frame: el equipo del jugador más cercano al balón, solo si está a <= R_CTRL m,
     el balón no va demasiado rápido (no es un pase en vuelo) y el rival no está igual de cerca
     (si los dos están encima es un balón disputado -> nadie).
  2. Toques: tramos seguidos de un mismo candidato (se toleran micro-huecos) de al menos
     MIN_TOUCH frames.
  3. Control confirmado del rival si: un toque largo (>= HOLD frames, "se queda el balón") o
     >= 2 toques suyos seguidos sin que el dueño actual toque en medio ("completa un pase").
     Los toques más viejos que CHAIN_S segundos se olvidan.
  4. El cambio se fecha al INICIO del primer toque de esa cadena (no al confirmarlo), porque la
     métrica castiga mucho llegar tarde a la transición.
  5. Tras cada balón muerto, el dueño inicial es el equipo del primer toque (quien saca).
  6. Limpieza: una posesión < MIN_SPELL s encajonada entre dos del mismo equipo se absorbe.

Mientras el balón no se ve, se mantiene el último dueño (como dice el reglamento: en balón
suelto o disputado la posesión sigue con el último equipo que la tuvo).
"""

import numpy as np

from signals import FPS, runs

HOME, AWAY, NONE = 0, 1, -1

# Params (fijados a priori, ver TECHNICAL_DOC.md)
R_CTRL = 1.8          # m: distancia máxima jugador-balón para considerar que lo controla
CONTEST_GAP = 0.75    # m: si el rival está a menos de esto extra, es balón disputado
MAX_CTRL_SPEED = 9.0  # m/s: más rápido que esto es un pase/tiro en vuelo, no control
CLOSE_GAP = 3         # frames: micro-huecos tolerados dentro de un toque
MIN_TOUCH = 4         # frames: duración mínima de un toque (0.16 s)
HOLD = 25             # frames: un toque de >= 1 s ya es control
CHAIN_S = 6.0         # s: memoria de la cadena de toques del rival
MIN_SPELL = 1.0       # s


def candidates(S):
    d_h, d_a = S["d_home"], S["d_away"]
    speed = np.nan_to_num(S["ball_speed"], nan=0.0)
    cand = np.full(S["n"], NONE, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        dh = np.where(np.isfinite(d_h), d_h, np.inf)
        da = np.where(np.isfinite(d_a), d_a, np.inf)
        ok = S["known"] & (speed <= MAX_CTRL_SPEED)
        home = ok & (dh <= R_CTRL) & (da - dh >= CONTEST_GAP)
        away = ok & (da <= R_CTRL) & (dh - da >= CONTEST_GAP)
    cand[home] = HOME
    cand[away] = AWAY
    return cand


def touches(cand, a, b):
    """Toques [(team, inicio, fin)] dentro de [a, b)."""
    c = cand[a:b].copy()
    # cerrar micro-huecos NONE entre dos frames del mismo equipo
    for team in (HOME, AWAY):
        for s, e in runs(c != team):
            if 0 < s and e < len(c) and (e - s) <= CLOSE_GAP and c[s - 1] == team and c[e] == team:
                if np.all(c[s:e] == NONE):
                    c[s:e] = team
    out = []
    for team in (HOME, AWAY):
        for s, e in runs(c == team):
            if e - s >= MIN_TOUCH:
                out.append((team, a + s, a + e))
    return sorted(out, key=lambda t: t[1])


def segment_owner(S, cand, a, b, fallback):
    """Etiquetas H/A para un tramo vivo [a, b)."""
    lab = np.empty(b - a, dtype=np.int8)
    ev = touches(cand, a, b)
    if not ev:
        seg = cand[a:b]
        nh, na = int(np.sum(seg == HOME)), int(np.sum(seg == AWAY))
        lab[:] = fallback if nh == na else (HOME if nh > na else AWAY)
        return lab

    owner = ev[0][0]               # quien saca / primer toque
    changes = [(a, owner)]
    chain = []                     # toques del rival desde el último toque del dueño
    for team, s, e in ev:
        if team == owner:
            chain = []
            continue
        chain = [t for t in chain if t[1] >= s - CHAIN_S * FPS] + [(team, s, e)]
        held = (e - s) >= HOLD
        if held or len(chain) >= 2:
            owner = team
            changes.append((chain[0][1], owner))
            chain = []

    for i, (start, team) in enumerate(changes):
        end = changes[i + 1][0] if i + 1 < len(changes) else b
        lab[start - a:end - a] = team
    return lab


def clean_short_spells(lab):
    """Absorbe posesiones < MIN_SPELL s encajonadas entre dos del mismo equipo."""
    min_len = int(MIN_SPELL * FPS)
    changed = True
    while changed:
        changed = False
        bounds = np.flatnonzero(np.diff(lab)) + 1
        edges = np.concatenate(([0], bounds, [len(lab)]))
        for i in range(1, len(edges) - 2):
            s, e = edges[i], edges[i + 1]
            if e - s < min_len and lab[s - 1] == lab[e]:
                lab[s:e] = lab[s - 1]
                changed = True
                break
    return lab


def assign(S, alive):
    """Array por frame: 0 = H, 1 = A, -1 = sin etiqueta (balón muerto o fuera de juego)."""
    cand = candidates(S)
    out = np.full(S["n"], NONE, dtype=np.int8)
    last = HOME
    for a, b in runs(alive):
        lab = clean_short_spells(segment_owner(S, cand, a, b, last))
        out[a:b] = lab
        last = int(lab[-1])
    return out
