# More than a Hack 2027 — Phase 1: Ball Possession
## Technical document — team DreamTeam

Repository: https://github.com/mikyEsanino/mtah2027-DreamTeam

---

## 1. Approach

We built a **rule-based, two-stage system with no learned parameters**:

1. **Dead-ball detector** (`src/dead_ball.py`) decides, frame by frame, whether the ball is out of
   play (`D`).
2. **Possession state machine** (`src/possession.py`) runs only on the live-ball stretches and
   decides which team is in possession (`H` / `A`).
3. **Integration** (`src/combine.py`) writes `X` outside the two graded halves, `D` where the
   dead-ball detector fires, and `H`/`A` elsewhere.

We chose rules over a learned model for three reasons:

- The ranking metric (`spellF1`) rewards **whole possessions with accurate boundaries**, not
  per-frame accuracy. A state machine with explicit "possession only changes on controlled
  possession" logic produces possession-shaped output by construction instead of a flickering
  per-frame classifier that then needs smoothing.
- We had **no labelled data on other games**, and the rules forbid training or tuning on the
  provided game or its public 5-minute ground truth.
- Every decision is explainable and maps directly to the challenge's own possession definition.

## 2. Assets used

| Asset | Used? | How |
|---|---|---|
| Tactical ATD (tracking + ball) | **Yes** | Ball and player positions (metres), player speeds |
| Smart Tagging | **Yes, partially** | Only `SET PIECES` instances, as *restart* anchors (kick-offs, throw-ins, corners, goal kicks, free kicks). Phase tags and team tags are not used |
| Half boundaries (`games.half_video_blocks`) | **Yes** | Published boundaries of the graded halves; frames outside are written as `X` |
| Homography | No | Not needed: the ATD is already in pitch coordinates |
| Tactical / broadcast video, broadcast ATD | No | Not enough time to align the broadcast timeline or build video models |
| Public 5-minute ground truth | **Only as a pipeline check** | Run once with `run_evaluation.py` to confirm frame alignment and file format. Never used to choose or tune parameters |

## 3. Training data

**None.** No model is trained. All thresholds are fixed *a priori* from football domain knowledge
(typical player and ball speeds, control distances, durations of stoppages) and are listed in the
`# Params` block at the top of each module. No hand labels of any game were produced or used.

## 4. Signals (`src/signals.py`)

Per video frame:

- **Ball position**, with gaps of ≤ 0.5 s (12 frames) linearly interpolated; longer gaps stay
  unknown. The ball is missing in ~41 % of graded frames, so we never assume it exists.
- **Ball speed**, from consecutive positions, 5-frame moving mean.
- **Ball out of pitch**: raw (non-interpolated) ball more than 1.0 m beyond a line
  (pitch 105.3 × 68.0 m).
- **Activity**: mean speed of all tracked players, 2-second moving mean.
- **Nearest home / nearest away player distance** to the ball. Track slots are not identities,
  so we only ever use team membership, never individual tracks.

## 5. Possession rules implemented

### Dead ball (`D`)

| Rule | Description |
|---|---|
| R1 Kick-off | Inside each half, everything before the kick-off is `D`. The kick-off is located with the `Kick Off (Start)` tag and refined to the first frame in its ±1 s-padded window where the ball speed exceeds 3 m/s for 3 frames |
| R2 Ball out | From the first frame the ball is out (≥ 0.2 s continuously) until the restart: the next Smart Tagging set piece within 90 s, refined the same way as R1. If there is none, until the ball is back inside the pitch and moving > 3 m/s (cap 60 s) |
| R3 Stoppage without the ball leaving the pitch (fouls, free kicks) | Before every set-piece restart not already preceded by `D`, mark `D` back to the last moment the 2-s player activity was ≥ 1.6 m/s (look back ≤ 45 s, minimum 2 s) |
| R4 Long stoppage | Ball invisible ≥ 3 s **and** mean activity < 1.2 m/s → `D` |
| R5 Clean-up | Live gaps < 1 s between dead stretches become `D`; dead stretches < 1 s are removed |

### Possession (`H` / `A`)

Following the challenge definition — *possession changes only when the opponent gains controlled
possession; touches, deflections and won aerial duels do not count; a loose or contested ball stays
with the last team*:

1. **Candidate per frame**: the team of the nearest player if that player is ≤ 1.8 m from the ball,
   the ball moves ≤ 9 m/s (a faster ball is a pass or shot in flight, not under control), and the
   nearest opponent is at least 0.75 m further away (otherwise the ball is *contested* → no
   candidate).
2. **Touches**: runs of the same candidate team of ≥ 4 frames (0.16 s), tolerating ≤ 3-frame
   dropouts.
3. **Controlled possession by the opponent** is confirmed when either
   - one touch lasts ≥ 1 s (*"a player keeps the ball"*), or
   - the opponent makes ≥ 2 consecutive touches with no touch by the current owner in between,
     within 6 s (*"a team completes a pass"*).
   A single short touch (deflection, block, header) never changes possession.
4. **Change timing**: the change is dated at the **start of the first touch of the confirming
   chain**, not at the moment of confirmation, because `spellF1` penalises late transitions
   heavily (1 s late ≈ −35 %).
5. **After every stoppage** the initial owner is the team of the first touch (the restart taker).
6. **Ball not visible**: the current owner is kept (frames are labelled, never `X`, inside the
   graded halves).
7. **Clean-up**: a possession shorter than 1 s sandwiched between two possessions of the same team
   is absorbed.

## 6. Limitations and what we would do with more time

- **Thresholds are not validated.** They are physically motivated but untested on any labelled
  game. With more time we would tune them on external open tracking data with event-derived
  possession (e.g. Metrica's public sample games), never on the provided match.
- **Missing ball (41 %)**: during long occlusions we keep the last owner, so possession changes
  inside an occlusion are detected late, at the next visible touch. A ball-trajectory model or the
  video would help.
- **Restart tags are incomplete** (only 59 set pieces, throw-ins without team). Stoppages missed by
  both the ball-out rule and the tags (e.g. some fouls) are only caught by R4.
- **Tracking noise**: spare balls near the touchline or ball misdetections can trigger false
  "ball out" events; duplicated or swapped tracks can create false touches.
- **Unused assets**: video, homography and the broadcast feed could recover the ball when the
  tactical tracking loses it.

## 7. Reproduction

Requirements: Python 3.12, `numpy==2.5.4`, `pandas==3.1.1` (`requirements-pipeline.txt`).

```bash
git clone https://github.com/mikyEsanino/mtah2027-DreamTeam.git
cd mtah2027-DreamTeam
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements-pipeline.txt
# put the provided data in data/tactical/ (see DATA.md); the video is not needed
python make_submission.py --team TEAM_NAME --rebuild-cache
python evaluation/validate_submission.py evaluation/results/TEAM_NAME.csv
```

The pipeline is deterministic: the same data produces the same file byte for byte. A cached copy of
the parsed tracking is written to `data/cache/` (git-ignored, never committed).

## 8. Note on tools

Parts of the code were written with the help of an AI coding assistant. In line with the terms of
the competition, **no match data (tracking, video, tagging or any extract of them) was shared with
any AI service**; the code was developed and tested on synthetic data and run on the real data only
locally.
