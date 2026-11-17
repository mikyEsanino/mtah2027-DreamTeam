"""Genera la submission completa desde la data cruda. Un solo comando:

    python make_submission.py --team NOMBRE_EQUIPO

Escribe evaluation/results/NOMBRE_EQUIPO.csv. Es determinista: misma data -> mismo archivo.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from atd_arrays import load_arrays  # noqa: E402
import signals  # noqa: E402
import dead_ball  # noqa: E402
import combine  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--team", required=True, help="nombre del equipo (nombre del CSV)")
    p.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    p.add_argument("--rebuild-cache", action="store_true",
                   help="vuelve a leer el ATD crudo en vez de usar data/cache/")
    args = p.parse_args()

    print("1/4 cargando tracking...")
    A = load_arrays(rebuild=args.rebuild_cache)
    print("2/4 calculando señales...")
    S = signals.compute(A)
    print("3/4 balón muerto...")
    restarts = dead_ball.load_restarts()
    dead, _ = dead_ball.detect(S, restarts)
    print("4/4 posesión e integración...")
    lab = combine.labels(S, dead)

    path = combine.write_csv(os.path.join(args.out, "{}.csv".format(args.team)), lab)
    counts = {k: int(np.sum(lab == k)) for k in "HADX"}
    print("\nlisto -> {}".format(os.path.relpath(path, ROOT)))
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("\nvalida antes de subir:\n  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
