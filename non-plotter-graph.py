# pip install pyserial pyqtgraph PyQt5
"""
Serial sensor visualizer — gesture & XY position detector.
Channels: PA0=right (red), PA5=left (green), PA6=middle (blue).

XY Estimation — superposition / basis-vector approach:
  Each calibrated axis anchor (LEFT, RIGHT, TOP, BOTTOM) is a "basis vector"
  in sensor space.  negative_drop_activation() measures how strongly the
  current signal projects onto each one.

    x_raw = right_act - left_act      (positive → finger is right of centre)
    y_raw = top_act   - bottom_act    (positive → finger is above centre)

  Dividing by (right_act + left_act) removes common-mode drift, so pure
  X motion produces a clean X signal with no Y bleed-through.

  Cross-axis softening (not hard suppression):
    ratio = weaker_axis_act / stronger_axis_act
    if ratio > CROSSAXIS_KEEP_RATIO   → both axes expressed fully (corner zone)
    if ratio < CROSSAXIS_ZERO_RATIO   → weaker axis zeroed (pure-axis zone)
    linear blend in between
    if BOTH axes exceed CORNER_ACTIVATION_LEVEL → never suppress (diagonal motion)

  A gamma response curve is applied for centre-sensitivity tuning.
  When all 9 position anchors are calibrated, k-NN weighted interpolation
  over the full grid is used instead for higher accuracy.
"""

import sys
import math
import serial
from collections import deque
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# ============================================================
# Hardware / serial settings
# ============================================================
PORT = "COM3"
BAUD = 38400
MAX_POINTS = 500            # samples shown in waveform plot
MAX_LINES_PER_UPDATE = 30   # cap on serial lines processed per timer tick

# ============================================================
# Smoothing windows
# ============================================================
SMOOTH_N = 5            # samples averaged for gesture classification
POSITION_SMOOTH_N = 2   # samples averaged for position (faster response)

# ============================================================
# Gesture detection thresholds
# ============================================================
PRESS_GATE_SUM = 25          # min channel sum to consider PRESS
NEGATIVE_GATE_MIN = -40      # at least one channel must fall below this for edge gestures
SIMILARITY_THRESHOLD = 0.90  # cosine similarity required to confirm a gesture
CLASSIFY_CONFIRM_SAMPLES = 2  # consecutive hits needed to fire onset
RELEASE_CONFIRM_SAMPLES = 3   # consecutive non-hits needed to fire release
PRESS_RELEASE_SUM = 15        # channel sum below this → PRESS released
NEGATIVE_RELEASE_LEVEL = -20  # all channels above this → edge gesture released

# ============================================================
# Calibration workflow
# ============================================================
CALIBRATION_SETTLE_SAMPLES = 8    # samples discarded at calibration start (let signal stabilise)
CALIBRATION_CAPTURE_SAMPLES = 25  # samples averaged to form the template

# ============================================================
# XY position estimation
# ============================================================
X_MAX_MM = 170.0   # half-range of X axis in mm
Y_MAX_MM = 170.0   # half-range of Y axis in mm

POSITION_IDLE_LEVEL = 0.006    # total axis activation below this → treat as idle
POSITION_ACTIVE_THRESHOLD = 0.15  # max(neg_act, pos_act) must exceed this to compute that axis
                                   # raise if Y drifts when centred; lower if edges feel sluggish
POSITION_DIFF_FLOOR = 0.002    # normalised differential below this → zero (deadband)
POSITION_RESPONSE_GAMMA = 0.65 # < 1 expands near-centre sensitivity; > 1 compresses it
POSITION_INTERP_POWER = 3.0    # k-NN exponent: higher → sharper localisation
POSITION_K_NEAREST = 4         # anchors used in k-NN interpolation

# Cross-axis softening — controls whether corners express both X and Y
# Raise CROSSAXIS_KEEP_RATIO → fewer corners detected, cleaner axis lines
# Lower CROSSAXIS_ZERO_RATIO → less axis bleed-through on straight motion
# Lower CORNER_ACTIVATION_LEVEL → corners activate sooner (may add diagonal noise)
CROSSAXIS_KEEP_RATIO = 0.70       # ratio above which both axes pass through fully
CROSSAXIS_ZERO_RATIO = 0.15       # ratio below which the weaker axis is zeroed
CORNER_ACTIVATION_LEVEL = 0.12    # if both axes exceed this, suppress nothing (corner)

# ============================================================
# Auto edge detection
# ============================================================
AUTO_EDGE_MIN_DROP = 60.0          # absolute drop required before attempting detection
AUTO_EDGE_MIN_SIMILARITY = 0.78    # min cosine similarity for a clean match
AUTO_EDGE_MIN_MARGIN = 0.03        # gap between 1st and 2nd match required

# Idealised relative drop patterns (PA0=right, PA5=left, PA6=middle).
# Tune these if auto-detect keeps mis-classifying:
#   Set AUTO_EDGE_MIN_SIMILARITY = 0.0, trigger auto-detect while holding each
#   edge, and note the printed per-edge similarity scores + raw drops.
AUTO_EDGE_PATTERNS = {
    "LEFT":   (0.15, 1.00, 0.70),  # left channel drops most
    "RIGHT":  (1.00, 0.15, 0.70),  # right channel drops most
    "TOP":    (0.80, 0.80, 1.00),  # all drop roughly equally, middle slightly more
    "BOTTOM": (0.50, 0.50, 1.00),  # all drop, middle dominates, sides moderate
}

# ============================================================
# Display
# ============================================================
DEFAULT_Y_MIN = -450
DEFAULT_Y_MAX = 150
Y_MARGIN = 20
POSITION_TRAIL_POINTS = 200

# ============================================================
# Names / coordinate mapping
# ============================================================
GESTURES = ["PRESS", "LEFT", "RIGHT", "TOP", "BOTTOM"]
EDGE_NAMES = ["LEFT", "RIGHT", "TOP", "BOTTOM"]
AUTO_EDGE_TARGET = "AUTO_EDGE"
POSITION_NAMES = [
    "CENTER",
    "LEFT", "RIGHT",
    "TOP", "BOTTOM",
    "TOP_LEFT", "TOP_RIGHT",
    "BOTTOM_LEFT", "BOTTOM_RIGHT",
]
POSITION_REQUIRED_NAMES = {"LEFT", "RIGHT", "TOP", "BOTTOM"}  # CENTER not required by estimator
POSITION_ANCHOR_COORDS = {
    "CENTER":       ( 0.0,       0.0      ),
    "LEFT":         (-X_MAX_MM,  0.0      ),
    "RIGHT":        ( X_MAX_MM,  0.0      ),
    "TOP":          ( 0.0,       Y_MAX_MM ),
    "BOTTOM":       ( 0.0,      -Y_MAX_MM ),
    "TOP_LEFT":     (-X_MAX_MM,  Y_MAX_MM ),
    "TOP_RIGHT":    ( X_MAX_MM,  Y_MAX_MM ),
    "BOTTOM_LEFT":  (-X_MAX_MM, -Y_MAX_MM ),
    "BOTTOM_RIGHT": ( X_MAX_MM, -Y_MAX_MM ),
}

# ============================================================
# Serial
# ============================================================
ser = serial.Serial(PORT, BAUD, timeout=0)

# ============================================================
# Runtime state
# ============================================================
baseline_pa0 = None
baseline_pa5 = None
baseline_pa6 = None
serial_buffer = ""  # renamed from 'buffer' to avoid shadowing the built-in

y_pa0 = deque(maxlen=MAX_POINTS)
y_pa5 = deque(maxlen=MAX_POINTS)
y_pa6 = deque(maxlen=MAX_POINTS)
position_x_history = deque(maxlen=POSITION_TRAIL_POINTS)
position_y_history = deque(maxlen=POSITION_TRAIL_POINTS)

sample_index = 0
gesture_state = "IDLE"
candidate_counts = {g: 0 for g in GESTURES}
release_count = 0

# Calibration templates — these persist across baseline resets
templates = {g: None for g in GESTURES}
position_templates = {name: None for name in POSITION_NAMES}

# Active calibration session (cleared on completion, not on baseline reset)
calibration_target = None
calibration_vectors = []
calibration_settle_remaining = 0

# ============================================================
# UI construction
# ============================================================
app = QtWidgets.QApplication(sys.argv)
win = QtWidgets.QWidget()
main_layout = QtWidgets.QVBoxLayout()
win.setLayout(main_layout)

# Waveform plot
plot = pg.PlotWidget()
plot.setTitle("Sensor Values")
plot.setLabel("left", "Value (baseline-subtracted)")
plot.setLabel("bottom", "Sample in window")
plot.showGrid(x=True, y=True)
plot.setYRange(DEFAULT_Y_MIN, DEFAULT_Y_MAX)
plot.setXRange(0, MAX_POINTS - 1)
plot.setMouseEnabled(x=False, y=False)
plot.setMenuEnabled(False)
plot.addLegend()

curve_pa0 = plot.plot(pen="r", name="PA0 right")
curve_pa5 = plot.plot(pen="g", name="PA5 left")
curve_pa6 = plot.plot(pen="b", name="PA6 middle")

# XY displacement plot
xy_plot = pg.PlotWidget()
xy_plot.setTitle("XY Displacement")
xy_plot.setLabel("left", "Y displacement (mm)")
xy_plot.setLabel("bottom", "X displacement (mm)")
xy_plot.showGrid(x=True, y=True)
xy_plot.setAspectLocked(True)
xy_plot.setMouseEnabled(x=False, y=False)
xy_plot.setMenuEnabled(False)
xy_plot.setXRange(-X_MAX_MM, X_MAX_MM, padding=0.05)
xy_plot.setYRange(-Y_MAX_MM, Y_MAX_MM, padding=0.05)
xy_plot.addLine(x=0, pen=pg.mkPen((180, 180, 180), width=1))
xy_plot.addLine(y=0, pen=pg.mkPen((180, 180, 180), width=1))
xy_trail_curve = xy_plot.plot(pen=pg.mkPen((255, 215, 0), width=2))
xy_point = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(255, 255, 0),
                               pen=pg.mkPen("w", width=1))
xy_plot.addItem(xy_point)

# Status labels
status_label         = QtWidgets.QLabel("State: IDLE")
score_label          = QtWidgets.QLabel("Best match: none")
template_label       = QtWidgets.QLabel("Gesture templates: none")
position_anchor_label = QtWidgets.QLabel("Position anchors: none")
calib_label          = QtWidgets.QLabel(
    "Calibration: baseline first, then calibrate PRESS and the 9 position anchors"
)
position_label       = QtWidgets.QLabel("Displacement: x=0.00 mm  y=0.00 mm")

main_layout.addWidget(plot)
main_layout.addWidget(xy_plot)
main_layout.addWidget(status_label)
main_layout.addWidget(score_label)
main_layout.addWidget(template_label)
main_layout.addWidget(position_anchor_label)
main_layout.addWidget(calib_label)
main_layout.addWidget(position_label)

# Buttons
row1 = QtWidgets.QHBoxLayout()
row2 = QtWidgets.QHBoxLayout()
row3 = QtWidgets.QHBoxLayout()

btn_rebaseline       = QtWidgets.QPushButton("Recalibrate Baseline")
btn_clear_templates  = QtWidgets.QPushButton("Clear Calibrations")
btn_cal_press        = QtWidgets.QPushButton("Calibrate PRESS")
btn_cal_left         = QtWidgets.QPushButton("Calibrate LEFT")
btn_cal_right        = QtWidgets.QPushButton("Calibrate RIGHT")
btn_cal_top          = QtWidgets.QPushButton("Calibrate TOP")
btn_cal_bottom       = QtWidgets.QPushButton("Calibrate BOTTOM")
btn_cal_auto_edge    = QtWidgets.QPushButton("Auto Detect Edge")
btn_cal_center       = QtWidgets.QPushButton("Calibrate CENTER")
btn_cal_top_left     = QtWidgets.QPushButton("Calibrate TOP_LEFT")
btn_cal_top_right    = QtWidgets.QPushButton("Calibrate TOP_RIGHT")
btn_cal_bottom_left  = QtWidgets.QPushButton("Calibrate BOTTOM_LEFT")
btn_cal_bottom_right = QtWidgets.QPushButton("Calibrate BOTTOM_RIGHT")

row1.addWidget(btn_rebaseline)
row1.addWidget(btn_clear_templates)
row2.addWidget(btn_cal_press)
row2.addWidget(btn_cal_left)
row2.addWidget(btn_cal_right)
row2.addWidget(btn_cal_top)
row2.addWidget(btn_cal_bottom)
row2.addWidget(btn_cal_auto_edge)
row3.addWidget(btn_cal_center)
row3.addWidget(btn_cal_top_left)
row3.addWidget(btn_cal_top_right)
row3.addWidget(btn_cal_bottom_left)
row3.addWidget(btn_cal_bottom_right)

main_layout.addLayout(row1)
main_layout.addLayout(row2)
main_layout.addLayout(row3)


# ============================================================
# Math helpers
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def vec_norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def cosine_similarity(a, b):
    na, nb = vec_norm(a), vec_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) / (na * nb)


def safe_similarity(v, t):
    if v is None or t is None:
        return None
    return max(0.0, cosine_similarity(v, t))


def vec_mean(vectors):
    n = len(vectors)
    if n == 0:
        return None
    return tuple(sum(v[i] for v in vectors) / n for i in range(3))


def moving_avg(values, n):
    vals = list(values)
    if not vals:
        return 0.0
    if len(vals) < n:
        return sum(vals) / len(vals)
    return sum(vals[-n:]) / n


def get_smoothed(n=SMOOTH_N):
    if len(y_pa0) < n:
        return None
    return (moving_avg(y_pa0, n), moving_avg(y_pa5, n), moving_avg(y_pa6, n))


# ============================================================
# UI helpers
# ============================================================

def clear_position_plot():
    position_x_history.clear()
    position_y_history.clear()
    xy_trail_curve.setData([], [])
    xy_point.setData([0.0], [0.0])


def update_template_label():
    gesture_done = [g for g in GESTURES if templates[g] is not None]
    anchor_done  = [name for name in POSITION_NAMES if position_templates[name] is not None]
    template_label.setText(
        "Gesture templates: " + (" | ".join(gesture_done) if gesture_done else "none")
    )
    position_anchor_label.setText(
        f"Position anchors ({len(anchor_done)}/{len(POSITION_NAMES)}): "
        + (" | ".join(anchor_done) if anchor_done else "none")
    )


def maybe_expand_ylim(v0, v5, v6):
    ymin, ymax = plot.getViewBox().viewRange()[1]
    low  = min(v0, v5, v6)
    high = max(v0, v5, v6)
    new_ymin = min(ymin, min(DEFAULT_Y_MIN, low  - Y_MARGIN)) if low  < ymin else ymin
    new_ymax = max(ymax, max(DEFAULT_Y_MAX, high + Y_MARGIN)) if high > ymax else ymax
    if new_ymin != ymin or new_ymax != ymax:
        plot.setYRange(new_ymin, new_ymax, padding=0)


# ============================================================
# Activation measurement
# ============================================================

def negative_drop_activation(v, template, weights=None):
    """
    Measure how closely v matches the negative-drop pattern of template.
    Returns ~1.0 for a full-magnitude match, 0.0 for no match.
    Only channels where template goes negative contribute.
    """
    eps = 1e-6
    if weights is None:
        weights = [1.0] * len(template)
    num = den = 0.0
    for value, ref, w in zip(v, template, weights):
        ref_drop = max(0.0, -ref)
        if ref_drop <= eps or w <= eps:
            continue
        num += max(0.0, -value) * ref_drop * w
        den += ref_drop * ref_drop * w
    return 0.0 if den <= eps else clamp(num / den, 0.0, 1.5)


def axis_channel_weights(neg_template, pos_template):
    """
    Per-channel weights that up-weight channels which differ most between
    the two opposite-edge templates, suppressing common-mode channels.
    """
    eps = 1e-6
    WEIGHT_POWER = 1.35
    neg_drops = [max(0.0, -x) for x in neg_template]
    pos_drops = [max(0.0, -x) for x in pos_template]
    diffs = [abs(p - n) for n, p in zip(neg_drops, pos_drops)]
    max_diff = max(diffs, default=0.0)
    if max_diff <= eps:
        return [1.0] * len(neg_template)
    return [(d / max_diff) ** WEIGHT_POWER for d in diffs]


# ============================================================
# XY position estimation — superposition / basis-vector
# ============================================================

def _axis_activation_pair(v, neg_name, pos_name):
    """Return (neg_act, pos_act) for one axis. Returns (0,0) if uncalibrated."""
    t_neg = position_templates.get(neg_name)
    t_pos = position_templates.get(pos_name)
    if t_neg is None or t_pos is None:
        return 0.0, 0.0
    weights = axis_channel_weights(t_neg, t_pos)
    return (
        negative_drop_activation(v, t_neg, weights),
        negative_drop_activation(v, t_pos, weights),
    )


def _soft_crossaxis(primary_act, secondary_act, primary_val, secondary_val):
    """
    Softly suppress the weaker axis.
    Both axes are in a corner → pass both through fully.
    One axis clearly dominates → suppress the other.
    Linear blend in the transition zone.
    """
    eps = 1e-6
    # Corner: both axes are meaningfully active → never suppress
    if primary_act >= CORNER_ACTIVATION_LEVEL and secondary_act >= CORNER_ACTIVATION_LEVEL:
        return primary_val, secondary_val

    ratio = secondary_act / max(primary_act, eps)
    keep = clamp(
        (ratio - CROSSAXIS_ZERO_RATIO) / max(CROSSAXIS_KEEP_RATIO - CROSSAXIS_ZERO_RATIO, eps),
        0.0, 1.0,
    )
    return primary_val, secondary_val * keep


def estimate_position_from_axis_templates(v):
    """
    Superposition / basis-vector XY estimator.
    Requires LEFT, RIGHT, TOP, BOTTOM templates at minimum.
    """
    eps = 1e-6
    required = ("LEFT", "RIGHT", "TOP", "BOTTOM")
    if any(position_templates.get(n) is None for n in required):
        return None

    left_act,   right_act = _axis_activation_pair(v, "LEFT",   "RIGHT")
    bottom_act, top_act   = _axis_activation_pair(v, "BOTTOM", "TOP")

    x_total = left_act  + right_act
    y_total = bottom_act + top_act

    # Raw differential axis signals.
    # We do NOT normalise by the total here — dividing by (left+right) would
    # strip out magnitude, making a small displacement look identical to a full
    # one (both give ±1.0).  Instead we use the raw difference so that a finger
    # halfway to the right edge gives ~half the output of full-right.
    #
    # axis_channel_weights already de-emphasises channels that are common to
    # both opposing templates, so cross-talk is handled at the activation level.
    #
    # Gate: require the dominant activation on each axis to exceed
    # POSITION_ACTIVE_THRESHOLD before computing that axis.
    # This prevents ADC noise (~±5 counts) from producing spurious displacement
    # when the sensor is at rest / centred.
    x_norm = clamp(right_act - left_act, -1.0, 1.0) \
        if max(left_act, right_act) > POSITION_ACTIVE_THRESHOLD else 0.0
    y_norm = clamp(top_act - bottom_act, -1.0, 1.0) \
        if max(bottom_act, top_act) > POSITION_ACTIVE_THRESHOLD else 0.0

    # Cross-axis softening: preserve corners, clean up straight lines
    if abs(x_norm) >= abs(y_norm):
        x_norm, y_norm = _soft_crossaxis(x_total, y_total, x_norm, y_norm)
    else:
        y_norm, x_norm = _soft_crossaxis(y_total, x_total, y_norm, x_norm)

    # Dead band
    if abs(x_norm) <= POSITION_DIFF_FLOOR:
        x_norm = 0.0
    if abs(y_norm) <= POSITION_DIFF_FLOOR:
        y_norm = 0.0

    # Gamma response and scale to mm
    x_mm = math.copysign(abs(x_norm) ** POSITION_RESPONSE_GAMMA * X_MAX_MM, x_norm) if x_norm else 0.0
    y_mm = math.copysign(abs(y_norm) ** POSITION_RESPONSE_GAMMA * Y_MAX_MM, y_norm) if y_norm else 0.0

    return clamp(x_mm, -X_MAX_MM, X_MAX_MM), clamp(y_mm, -Y_MAX_MM, Y_MAX_MM)


def _vector_distance2(a, b, scales):
    return sum(((ai - bi) / s) ** 2 for ai, bi, s in zip(a, b, scales))


def estimate_position_from_full_grid(v):
    """
    k-NN weighted interpolation over all 9 calibrated anchors.
    Only active when all 9 are available; falls back to axis estimator otherwise.
    """
    eps = 1e-6
    anchors = [
        (position_templates[n], POSITION_ANCHOR_COORDS[n][0], POSITION_ANCHOR_COORDS[n][1])
        for n in POSITION_NAMES
        if position_templates[n] is not None
    ]
    if len(anchors) < len(POSITION_NAMES):
        return None

    activity = max(negative_drop_activation(v, vec) for vec, _, _ in anchors)
    if activity <= POSITION_IDLE_LEVEL:
        return 0.0, 0.0

    channel_scales = [
        max(max(vec[i] for vec, _, _ in anchors) - min(vec[i] for vec, _, _ in anchors), 1.0)
        for i in range(3)
    ]

    distances = []
    for vec, x_mm, y_mm in anchors:
        d2 = _vector_distance2(v, vec, channel_scales)
        if d2 <= eps:
            return x_mm, y_mm
        distances.append((d2, x_mm, y_mm))

    distances.sort(key=lambda item: item[0])
    nearest = distances[:POSITION_K_NEAREST]

    total_w = x_sum = y_sum = 0.0
    for d2, x_mm, y_mm in nearest:
        w = 1.0 / (d2 + eps) ** (POSITION_INTERP_POWER / 2.0)
        total_w += w
        x_sum   += w * x_mm
        y_sum   += w * y_mm

    if total_w <= eps:
        return 0.0, 0.0
    return clamp(x_sum / total_w, -X_MAX_MM, X_MAX_MM), clamp(y_sum / total_w, -Y_MAX_MM, Y_MAX_MM)


def estimate_position_mm(v):
    """
    Unified position estimator.
    Prefers full 9-anchor k-NN grid when available, falls back to superposition.
    """
    if v is None:
        return None
    full_grid = estimate_position_from_full_grid(v)
    if full_grid is not None:
        return full_grid
    return estimate_position_from_axis_templates(v)


def position_calibration_ready():
    return all(position_templates[n] is not None for n in POSITION_REQUIRED_NAMES)


# ============================================================
# Gesture classification
# ============================================================

def reset_counts():
    global release_count
    for g in GESTURES:
        candidate_counts[g] = 0
    release_count = 0


def classify_vector(v):
    cur0, cur5, cur6 = v
    s_total = cur0 + cur5 + cur6
    candidates = []

    if templates["PRESS"] is not None:
        t = templates["PRESS"]
        if (s_total >= PRESS_GATE_SUM
                and sum(1 for x in v if x > 0) >= 2
                and vec_norm(v) >= 0.35 * vec_norm(t)):
            candidates.append(("PRESS", cosine_similarity(v, t)))

    if min(cur0, cur5, cur6) <= NEGATIVE_GATE_MIN:
        for g in EDGE_NAMES:
            t = templates[g]
            if t is None:
                continue
            if vec_norm(v) >= 0.35 * vec_norm(t):
                candidates.append((g, cosine_similarity(v, t)))

    if not candidates:
        return None, 0.0
    best_name, best_sim = max(candidates, key=lambda x: x[1])
    return (best_name, best_sim) if best_sim >= SIMILARITY_THRESHOLD else (None, best_sim)


def should_release(v, state):
    cur0, cur5, cur6 = v
    if state == "PRESS":
        return (cur0 + cur5 + cur6) <= PRESS_RELEASE_SUM
    return (cur0 >= NEGATIVE_RELEASE_LEVEL
            and cur5 >= NEGATIVE_RELEASE_LEVEL
            and cur6 >= NEGATIVE_RELEASE_LEVEL)


def handle_gesture_state(v):
    global gesture_state, release_count

    if calibration_target is not None:
        update_calibration(v)
        return

    best_name, best_sim = classify_vector(v)
    score_label.setText(f"Best match: {best_name or 'none'} | similarity={best_sim:.3f}")

    if gesture_state == "IDLE":
        if best_name is None:
            reset_counts()
            return
        for g in GESTURES:
            candidate_counts[g] = candidate_counts[g] + 1 if g == best_name else 0
        if candidate_counts[best_name] >= CLASSIFY_CONFIRM_SAMPLES:
            gesture_state = best_name
            reset_counts()
            status_label.setText(f"State: {best_name}")
            print(f"{best_name} detected")
    else:
        if should_release(v, gesture_state):
            release_count += 1
            if release_count >= RELEASE_CONFIRM_SAMPLES:
                old_state = gesture_state
                gesture_state = "IDLE"
                release_count = 0
                status_label.setText("State: IDLE")
                print(f"{old_state} released")
        else:
            release_count = 0


# ============================================================
# Calibration
# ============================================================

def is_good_calibration_pose(target, v):
    """
    Gate check: only applied to PRESS and AUTO_EDGE where we need a specific
    signal shape.  All position calibrations (edges, corners, center) trust the
    user to be in the right position — the settle period handles transients.
    """
    cur0, cur5, cur6 = v
    if target == "PRESS":
        return (cur0 + cur5 + cur6) >= PRESS_GATE_SUM and sum(1 for x in v if x > 0) >= 2
    if target == AUTO_EDGE_TARGET:
        return min(cur0, cur5, cur6) <= NEGATIVE_GATE_MIN
    return True  # CENTER, LEFT, RIGHT, TOP, BOTTOM, corners — always capture


def detect_edge_name(v):
    """
    Classify signal v as one of the 4 edges using cosine similarity against
    AUTO_EDGE_PATTERNS.  Returns (edge_name, score) or (None, score).

    Tuning tip: set AUTO_EDGE_MIN_SIMILARITY = 0.0, hold each edge, trigger
    auto-detect, and read the printed scores.  Adjust AUTO_EDGE_PATTERNS so
    that each edge's true pattern has the highest similarity score.
    """
    drops = tuple(max(0.0, -x) for x in v)
    if max(drops) < AUTO_EDGE_MIN_DROP:
        return None, 0.0

    scores = [(name, cosine_similarity(drops, AUTO_EDGE_PATTERNS[name]))
              for name in EDGE_NAMES]
    scores.sort(key=lambda item: item[1], reverse=True)
    best_name, best_sim = scores[0]
    second_sim = scores[1][1] if len(scores) > 1 else 0.0

    print(f"Auto edge scores: {scores}")
    if best_sim < AUTO_EDGE_MIN_SIMILARITY or (best_sim - second_sim) < AUTO_EDGE_MIN_MARGIN:
        return None, best_sim
    return best_name, best_sim


def start_calibration(name):
    global calibration_target, calibration_vectors, calibration_settle_remaining
    global gesture_state
    calibration_target = name
    calibration_vectors = []
    calibration_settle_remaining = CALIBRATION_SETTLE_SAMPLES
    gesture_state = "IDLE"
    reset_counts()

    if name == AUTO_EDGE_TARGET:
        calib_label.setText(
            "Move to any edge (LEFT / RIGHT / TOP / BOTTOM) and hold still for auto-detect"
        )
        print("Auto edge calibration started. Move to an edge and hold still.")
    else:
        calib_label.setText(
            f"Move to {name}, hold still — settling ({CALIBRATION_SETTLE_SAMPLES} samples)..."
        )
        print(f"Calibration started for {name}.")


def update_calibration(v):
    global calibration_target, calibration_vectors, calibration_settle_remaining

    if calibration_target is None:
        return

    if calibration_settle_remaining > 0:
        calibration_settle_remaining -= 1
        done = CALIBRATION_SETTLE_SAMPLES - calibration_settle_remaining
        calib_label.setText(
            f"Settling for {calibration_target} ({done}/{CALIBRATION_SETTLE_SAMPLES})..."
        )
        return

    # Gate: wait for a plausible signal before capturing
    if not is_good_calibration_pose(calibration_target, v):
        calib_label.setText(
            f"Waiting for valid {calibration_target} signal "
            f"({len(calibration_vectors)}/{CALIBRATION_CAPTURE_SAMPLES}) — "
            + ("hold position" if calibration_target != "AUTO_EDGE" else "move to edge")
        )
        return

    calibration_vectors.append(v)
    calib_label.setText(
        f"Capturing {calibration_target} "
        f"({len(calibration_vectors)}/{CALIBRATION_CAPTURE_SAMPLES})"
    )

    if len(calibration_vectors) < CALIBRATION_CAPTURE_SAMPLES:
        return

    mean_v = vec_mean(calibration_vectors)

    if calibration_target == AUTO_EDGE_TARGET:
        detected_name, score = detect_edge_name(mean_v)
        if detected_name is None:
            print(f"Auto edge failed. Best similarity={score:.3f}. "
                  "See per-edge scores above. Adjust AUTO_EDGE_PATTERNS if needed.")
            calib_label.setText(
                f"Auto-detect failed (best similarity={score:.3f}). "
                "Hold a clearer edge position and try again."
            )
        else:
            templates[detected_name] = mean_v
            position_templates[detected_name] = mean_v
            print(f"Auto-calibrated {detected_name}: {mean_v} | similarity={score:.3f}")
            calib_label.setText(
                f"Auto-detected {detected_name} saved (similarity={score:.3f})"
            )
    else:
        if calibration_target in templates:
            templates[calibration_target] = mean_v
        if calibration_target in position_templates:
            position_templates[calibration_target] = mean_v
        print(f"Calibrated {calibration_target}: {mean_v}")
        calib_label.setText(f"{calibration_target} calibration saved.")

    calibration_target = None
    calibration_vectors = []
    calibration_settle_remaining = 0
    update_template_label()


# ============================================================
# Baseline reset — intentionally does NOT touch saved calibrations
# ============================================================

def recalibrate_baseline():
    """
    Reset only live session state: baseline values, signal buffers, gesture tracking.
    Saved gesture templates and position anchors are PRESERVED.
    To erase calibrations use 'Clear Calibrations'.
    """
    global baseline_pa0, baseline_pa5, baseline_pa6
    global sample_index, serial_buffer, gesture_state
    global calibration_target, calibration_vectors, calibration_settle_remaining

    # Live signal state
    baseline_pa0 = baseline_pa5 = baseline_pa6 = None
    sample_index = 0
    serial_buffer = ""
    gesture_state = "IDLE"

    # Cancel any in-progress calibration session (templates already saved are kept)
    calibration_target = None
    calibration_vectors = []
    calibration_settle_remaining = 0

    reset_counts()
    y_pa0.clear()
    y_pa5.clear()
    y_pa6.clear()
    clear_position_plot()
    curve_pa0.setData([], [])
    curve_pa5.setData([], [])
    curve_pa6.setData([], [])
    plot.setYRange(DEFAULT_Y_MIN, DEFAULT_Y_MAX, padding=0)

    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    status_label.setText("State: IDLE")
    score_label.setText("Best match: none")
    position_label.setText("Displacement: x=0.00 mm  y=0.00 mm")
    update_template_label()
    calib_label.setText(
        "Baseline reset — waiting for first valid sample. "
        "Gesture templates and position anchors were preserved."
    )
    print("Baseline reset. Next valid sample sets new baseline. Calibrations preserved.")


def clear_templates():
    """Erase all saved gesture templates and position anchors."""
    global templates, position_templates, calibration_target, calibration_vectors
    global calibration_settle_remaining
    templates = {g: None for g in GESTURES}
    position_templates = {name: None for name in POSITION_NAMES}
    calibration_target = None
    calibration_vectors = []
    calibration_settle_remaining = 0
    update_template_label()
    clear_position_plot()
    position_label.setText("Displacement: x=0.00 mm  y=0.00 mm")
    calib_label.setText("All gesture templates and position anchors cleared.")
    print("All calibrations cleared.")


# ============================================================
# Button wiring
# ============================================================
btn_rebaseline.clicked.connect(recalibrate_baseline)
btn_clear_templates.clicked.connect(clear_templates)
btn_cal_press.clicked.connect(lambda: start_calibration("PRESS"))
btn_cal_left.clicked.connect(lambda: start_calibration("LEFT"))
btn_cal_right.clicked.connect(lambda: start_calibration("RIGHT"))
btn_cal_top.clicked.connect(lambda: start_calibration("TOP"))
btn_cal_bottom.clicked.connect(lambda: start_calibration("BOTTOM"))
btn_cal_auto_edge.clicked.connect(lambda: start_calibration(AUTO_EDGE_TARGET))
btn_cal_center.clicked.connect(lambda: start_calibration("CENTER"))
btn_cal_top_left.clicked.connect(lambda: start_calibration("TOP_LEFT"))
btn_cal_top_right.clicked.connect(lambda: start_calibration("TOP_RIGHT"))
btn_cal_bottom_left.clicked.connect(lambda: start_calibration("BOTTOM_LEFT"))
btn_cal_bottom_right.clicked.connect(lambda: start_calibration("BOTTOM_RIGHT"))


# ============================================================
# Main update loop (driven by QTimer at 1 ms)
# ============================================================

def update():
    global serial_buffer, baseline_pa0, baseline_pa5, baseline_pa6, sample_index

    n = ser.in_waiting
    if n <= 0:
        return

    try:
        chunk = ser.read(n).decode(errors="ignore")
    except Exception:
        return
    if not chunk:
        return

    serial_buffer += chunk
    lines = serial_buffer.split("\n")
    serial_buffer = lines[-1]
    complete_lines = lines[:-1]

    if not complete_lines:
        return
    if len(complete_lines) > MAX_LINES_PER_UPDATE:
        complete_lines = complete_lines[-MAX_LINES_PER_UPDATE:]

    updated = False

    for line in complete_lines:
        line = line.strip()
        if not line:
            continue

        try:
            parts = line.split(",")
            raw_pa0 = int(parts[0].split(":")[1].strip())
            raw_pa5 = int(parts[1].split(":")[1].strip())
            raw_pa6 = int(parts[2].split(":")[1].strip())
        except (IndexError, ValueError):
            continue

        if baseline_pa0 is None:
            baseline_pa0, baseline_pa5, baseline_pa6 = raw_pa0, raw_pa5, raw_pa6
            print(f"Baseline set: PA0={baseline_pa0}, PA5={baseline_pa5}, PA6={baseline_pa6}")
            calib_label.setText(
                "Baseline ready. Calibrate PRESS and "
                "CENTER / LEFT / RIGHT / TOP / BOTTOM / corners."
            )
            continue

        val_pa0 = raw_pa0 - baseline_pa0
        val_pa5 = raw_pa5 - baseline_pa5
        val_pa6 = raw_pa6 - baseline_pa6

        y_pa0.append(val_pa0)
        y_pa5.append(val_pa5)
        y_pa6.append(val_pa6)
        updated = True
        sample_index += 1

        maybe_expand_ylim(val_pa0, val_pa5, val_pa6)

        gesture_v = get_smoothed(SMOOTH_N)
        if gesture_v is not None:
            handle_gesture_state(gesture_v)

        position_v = get_smoothed(POSITION_SMOOTH_N)
        if position_v is not None and position_calibration_ready():
            pos = estimate_position_mm(position_v)
            if pos is not None:
                x_mm, y_mm = pos
                position_label.setText(f"Displacement: x={x_mm:+.2f} mm  y={y_mm:+.2f} mm")
                position_x_history.append(x_mm)
                position_y_history.append(y_mm)
                xy_trail_curve.setData(list(position_x_history), list(position_y_history))
                xy_point.setData([x_mm], [y_mm])

    if updated:
        x = list(range(len(y_pa0)))
        curve_pa0.setData(x, list(y_pa0))
        curve_pa5.setData(x, list(y_pa5))
        curve_pa6.setData(x, list(y_pa6))


# ============================================================
# Start
# ============================================================
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

win.resize(1250, 1000)
win.show()

print(f"Reading {PORT} @ {BAUD} ... first valid sample sets baseline")

try:
    sys.exit(app.exec_())
finally:
    ser.close()
