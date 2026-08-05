import cv2
import numpy as np

# Modern overlay palette (BGR)
_PANEL_BG = (32, 30, 28)
_PANEL_BORDER = (70, 66, 62)
_TEXT_PRIMARY = (248, 246, 242)
_TEXT_MUTED = (170, 165, 158)
_BAR_TRACK = (58, 54, 50)
_FRAME_ACCENT = (196, 168, 92)

_AGE_ACCENT = (92, 168, 220)
_GENDER_ACCENT = (140, 168, 92)
_EMOTION_ACCENTS = {
    "Angry": (70, 70, 220),
    "Disgust": (90, 140, 90),
    "Fear": (160, 100, 180),
    "Happy": (70, 190, 210),
    "Sad": (200, 150, 90),
    "Surprise": (120, 180, 255),
    "Neutral": (160, 160, 160),
}

_EMOTION_ORDER = list(_EMOTION_ACCENTS.keys())


def load_haar_cascade(cascade_path):
    return cv2.CascadeClassifier(cascade_path)


def _blend_rect(frame, x1, y1, x2, y2, color, alpha):
    """Draw a filled rectangle with alpha blending."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_corner_brackets(frame, x, y, w, h, color, thickness=2, length=18):
    """Draw L-shaped corner brackets around a face instead of a full box."""
    length = min(length, w // 3, h // 3)
    cv2.line(frame, (x, y), (x + length, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x, y + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness, cv2.LINE_AA)


def _put_text(frame, text, org, scale, color, thickness=1):
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_label(frame, text, x, y):
    """Legacy single-line label (kept for compatibility)."""
    _put_text(frame, text, (x, y), 0.55, _TEXT_PRIMARY, 1)


def _panel_rect(face_box, panel_w, panel_h, frame_shape, occupied):
    """Pick a panel position near the face that stays on-screen and avoids overlap."""
    x, y, w, h = face_box
    fh, fw = frame_shape[:2]
    gap = 10

    candidates = [
        (x, y - panel_h - gap),                 # above
        (x, y + h + gap),                       # below
        (x + w + gap, y),                       # right
        (x - panel_w - gap, y),                 # left
        (x + w + gap, y + h - panel_h),         # right-bottom
        (x - panel_w - gap, y + h - panel_h),   # left-bottom
    ]

    def fits(px, py):
        return px >= 8 and py >= 8 and px + panel_w <= fw - 8 and py + panel_h <= fh - 8

    def overlaps(px, py):
        for ox, oy, ow, oh in occupied:
            if not (px + panel_w <= ox or ox + ow <= px or py + panel_h <= oy or oy + oh <= py):
                return True
        return False

    for px, py in candidates:
        if fits(px, py) and not overlaps(px, py):
            return px, py

    # Fall back to first on-screen candidate even if overlapping
    for px, py in candidates:
        if fits(px, py):
            return px, py

    return max(8, min(x, fw - panel_w - 8)), max(8, min(y, fh - panel_h - 8))


def draw_face_overlay(frame, x, y, w, h, gender, age, emotion, emotion_scores=None, occupied=None):
    """
    Modern face annotation: corner brackets plus a translucent
    info panel with age, gender, and per-emotion likelihood bars.
    Works for multiple faces; pass `occupied` to avoid panel stacking.
    """
    if occupied is None:
        occupied = []

    emotion_color = _EMOTION_ACCENTS.get(emotion, _FRAME_ACCENT)
    _draw_corner_brackets(frame, x, y, w, h, emotion_color, thickness=2, length=20)

    tick_w = max(24, w // 4)
    tick_x = x + (w - tick_w) // 2
    cv2.line(frame, (tick_x, y), (tick_x + tick_w, y), emotion_color, 2, cv2.LINE_AA)

    if emotion_scores is None:
        emotion_scores = {label: (1.0 if label == emotion else 0.0) for label in _EMOTION_ORDER}
    rows = [(label, emotion_scores.get(label, 0.0)) for label in _EMOTION_ORDER]

    meta_rows = [
        ("AGE", str(age), _AGE_ACCENT),
        ("GENDER", str(gender).upper(), _GENDER_ACCENT),
    ]

    pad_x, pad_y = 12, 10
    meta_row_h = 20
    emotion_row_h = 18
    section_gap = 8
    label_w = 70
    bar_w = 88
    pct_w = 38
    value_scale = 0.45
    label_scale = 0.36
    emotion_scale = 0.36
    pct_scale = 0.34

    panel_w = pad_x * 2 + label_w + bar_w + pct_w
    panel_h = (
        pad_y * 2
        + meta_row_h * len(meta_rows)
        + section_gap
        + 14
        + emotion_row_h * len(rows)
    )

    panel_x, panel_y = _panel_rect((x, y, w, h), panel_w, panel_h, frame.shape, occupied)
    occupied.append((panel_x, panel_y, panel_w, panel_h))

    _blend_rect(frame, panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, _PANEL_BG, 0.78)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        _PANEL_BORDER,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + 3, panel_y + panel_h),
        emotion_color,
        -1,
        cv2.LINE_AA,
    )

    for i, (label, value, accent) in enumerate(meta_rows):
        cy = panel_y + pad_y + i * meta_row_h + 12
        cv2.circle(frame, (panel_x + 14, cy - 4), 3, accent, -1, cv2.LINE_AA)
        _put_text(frame, label, (panel_x + 24, cy), label_scale, _TEXT_MUTED, 1)
        _put_text(frame, value, (panel_x + pad_x + label_w, cy), value_scale, _TEXT_PRIMARY, 1)

    section_y = panel_y + pad_y + meta_row_h * len(meta_rows) + section_gap
    _put_text(frame, "EMOTIONS", (panel_x + 12, section_y + 10), label_scale, _TEXT_MUTED, 1)

    bar_x = panel_x + pad_x + label_w
    for i, (label, score) in enumerate(rows):
        cy = section_y + 14 + i * emotion_row_h + 12
        accent = _EMOTION_ACCENTS.get(label, _FRAME_ACCENT)
        is_top = label == emotion
        name_color = _TEXT_PRIMARY if is_top else _TEXT_MUTED

        _put_text(frame, label.upper(), (panel_x + 12, cy), emotion_scale, name_color, 1)

        bar_y1 = cy - 9
        bar_y2 = cy - 3
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + bar_w, bar_y2), _BAR_TRACK, -1, cv2.LINE_AA)
        fill_w = max(0, int(round(bar_w * float(np.clip(score, 0.0, 1.0)))))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + fill_w, bar_y2), accent, -1, cv2.LINE_AA)

        pct = f"{int(round(score * 100))}%"
        _put_text(frame, pct, (bar_x + bar_w + 6, cy), pct_scale, name_color, 1)

    return occupied
