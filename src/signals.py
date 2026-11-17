"""Señales por frame derivadas del tracking. Las usan dead_ball.py y possession.py.

Todo en metros, segundos y frames de VIDEO (0..199318). Nada aquí mira el ground truth.
"""

import warnings

import numpy as np

FPS = 25

# Params (elegidos a priori; ver TECHNICAL_DOC.md)
MAX_BALL_GAP = 12      # frames: huecos del balón <= 0.5 s se interpolan linealmente
SPEED_SMOOTH = 5       # frames: media móvil de la velocidad del balón
ACTIVITY_WINDOW = 50   # frames: ventana (2 s) para la velocidad media de los jugadores
OUT_MARGIN = 1.0       # m: el balón cuenta como fuera si pasa la línea por más de esto


def moving_nanmean(x, k):
    """Media móvil centrada de ancho k que ignora NaN."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    kernel = np.ones(k)
    num = np.convolve(np.where(ok, x, 0.0), kernel, mode="same")
    den = np.convolve(ok.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def runs(mask):
    """[(inicio, fin)) de cada tramo True de un array booleano."""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))


def interpolate_ball(ball, max_gap=MAX_BALL_GAP):
    """Rellena linealmente huecos cortos (<= max_gap frames) entre dos observaciones."""
    out = ball.copy()
    seen = np.isfinite(ball[:, 0])
    for a, b in runs(~seen):
        if a == 0 or b >= len(ball) or (b - a) > max_gap:
            continue
        p0, p1 = ball[a - 1], ball[b]
        w = (np.arange(a, b) - (a - 1)) / float(b - (a - 1))
        out[a:b] = p0 + (p1 - p0) * w[:, None]
    return out


def player_speeds(players, speed):
    """Velocidad de cada slot. Usa la del ATD; si no viene, la calcula con la posición."""
    if np.isfinite(speed).any():
        return speed
    s = np.full(speed.shape, np.nan, dtype=np.float32)
    s[1:] = np.linalg.norm(np.diff(players, axis=0), axis=2) * FPS
    return s


def nearest_by_team(players, team, ball, chunk=20000):
    """Distancia (m) del jugador más cercano al balón, por equipo. NaN si no hay datos."""
    n = len(ball)
    d_home = np.full(n, np.nan, dtype=np.float32)
    d_away = np.full(n, np.nan, dtype=np.float32)
    home, away = team == 0, team == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            d = np.linalg.norm(players[a:b] - ball[a:b, None, :], axis=2)
            d_home[a:b] = np.nanmin(d[:, home], axis=1)
            d_away[a:b] = np.nanmin(d[:, away], axis=1)
    return d_home, d_away


def compute(A):
    """Diccionario de señales por frame a partir de los arrays de atd_arrays.load_arrays()."""
    L, W = [float(v) for v in A["pitch"]]
    ball_raw = A["ball"].astype(np.float64)
    visible = np.isfinite(ball_raw[:, 0])
    ball = interpolate_ball(ball_raw)
    known = np.isfinite(ball[:, 0])

    step = np.full(len(ball), np.nan)
    step[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    ball_speed = moving_nanmean(step, SPEED_SMOOTH)

    with np.errstate(invalid="ignore"):
        out = visible & ((ball_raw[:, 0] < -OUT_MARGIN) | (ball_raw[:, 0] > L + OUT_MARGIN) |
                         (ball_raw[:, 1] < -OUT_MARGIN) | (ball_raw[:, 1] > W + OUT_MARGIN))
        inside = known & (ball[:, 0] >= 0) & (ball[:, 0] <= L) & (ball[:, 1] >= 0) & (ball[:, 1] <= W)

    spd = player_speeds(A["players"], A["speed"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_speed = np.nanmean(spd, axis=1)
    activity = moving_nanmean(mean_speed, ACTIVITY_WINDOW)

    d_home, d_away = nearest_by_team(A["players"].astype(np.float32), A["team"],
                                     ball.astype(np.float32))
    return {
        "n": len(ball), "visible": visible, "known": known, "ball": ball,
        "ball_speed": ball_speed, "out": out, "inside": inside,
        "activity": activity, "d_home": d_home, "d_away": d_away,
    }
