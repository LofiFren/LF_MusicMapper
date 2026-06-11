"""
widgets.py - Custom-painted controls shared by mapper.py and effects.py.

MiniKnob is a drop-in QDial replacement with a flat, minimal look:
a 300-degree value arc, a thin pointer, and a subtle face. It keeps
QDial's mouse/keyboard interaction (which sweeps 300 degrees with a
60-degree gap at the bottom) so dragging matches the drawn arc.
"""

import math

try:
    from PySide2 import QtCore, QtWidgets, QtGui
except ImportError:
    from PyQt5 import QtCore, QtWidgets, QtGui


class MiniKnob(QtWidgets.QDial):
    """Flat minimal knob: value arc + pointer, replaces the default QDial look.

    Interaction is DAW-style, not QDial-style: clicking never jumps the
    value. Drag vertically with "resistance" (DRAG_RANGE_PX pixels for a
    full sweep), hold Shift for fine adjustment, double-click to reset to
    the default, scroll for small steps.
    """

    # Match QDial's internal non-wrapping geometry: 300-degree sweep,
    # 60-degree gap centered at the bottom. Qt angle convention:
    # 0 = 3 o'clock, counter-clockwise positive.
    SPAN_DEG = 300.0
    START_DEG = 240.0

    DRAG_RANGE_PX = 220.0   # vertical pixels for a full min→max sweep
    FINE_FACTOR = 8.0       # Shift slows the drag by this much
    WHEEL_STEPS = 100       # wheel notches for a full sweep

    def __init__(self, parent=None):
        super().__init__(parent)
        self._accent = QtGui.QColor('#4B9DE0')
        self._track = QtGui.QColor(127, 127, 127, 70)
        self._face = QtGui.QColor(0, 0, 0, 40)
        self._drag_anchor = None     # (y, value) at drag start / re-anchor
        self._drag_fine = False
        self._default_value = None
        self.setNotchesVisible(False)
        self.setToolTip("Drag up/down — Shift for fine, double-click to reset")

    def set_accent_color(self, color):
        """Set the value-arc / pointer color (e.g. the track color)."""
        self._accent = QtGui.QColor(color)
        self.update()

    def set_default_value(self, value):
        """Value restored on double-click."""
        self._default_value = int(value)

    # ── DAW-style interaction (replaces QDial's jump-to-angle) ──────────

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_fine = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
            self._drag_anchor = (event.pos().y(), self.value())
            self.setCursor(QtCore.Qt.SizeVerCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_anchor is None:
            super().mouseMoveEvent(event)
            return
        fine = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
        if fine != self._drag_fine:
            # Re-anchor when Shift is pressed/released mid-drag so the
            # value doesn't jump with the new sensitivity
            self._drag_fine = fine
            self._drag_anchor = (event.pos().y(), self.value())
        anchor_y, anchor_val = self._drag_anchor
        dy = anchor_y - event.pos().y()  # up = increase
        rng = self.maximum() - self.minimum()
        px = self.DRAG_RANGE_PX * (self.FINE_FACTOR if fine else 1.0)
        self.setValue(round(anchor_val + dy / px * rng))
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_anchor is not None:
            self._drag_anchor = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._default_value is not None:
            self.setValue(self._default_value)
        event.accept()

    def wheelEvent(self, event):
        rng = self.maximum() - self.minimum()
        step = max(1, round(rng / self.WHEEL_STEPS))
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            step = 1
        notches = event.angleDelta().y() / 120.0
        self.setValue(round(self.value() + notches * step))
        event.accept()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        w, h = self.width(), self.height()
        side = min(w, h)
        arc_w = max(2.5, side * 0.075)
        margin = arc_w / 2 + 1
        rect = QtCore.QRectF((w - side) / 2 + margin, (h - side) / 2 + margin,
                             side - 2 * margin, side - 2 * margin)

        rng = self.maximum() - self.minimum()
        frac = (self.value() - self.minimum()) / rng if rng else 0.0

        accent = QtGui.QColor(self._accent)
        if not self.isEnabled():
            accent.setAlpha(70)

        # Face disc
        face_rect = rect.adjusted(arc_w, arc_w, -arc_w, -arc_w)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(self._face)
        p.drawEllipse(face_rect)

        # Track arc (full sweep, dim)
        pen = QtGui.QPen(self._track, arc_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, int(self.START_DEG * 16), int(-self.SPAN_DEG * 16))

        # Value arc
        if frac > 0.0:
            pen.setColor(accent)
            p.setPen(pen)
            p.drawArc(rect, int(self.START_DEG * 16), int(-self.SPAN_DEG * frac * 16))

        # Pointer line (center of face toward current angle)
        angle = math.radians(self.START_DEG - self.SPAN_DEG * frac)
        cx, cy = rect.center().x(), rect.center().y()
        r_outer = face_rect.width() / 2 - max(1.0, arc_w * 0.3)
        r_inner = r_outer * 0.45
        pointer = QtGui.QPen(accent, max(1.5, arc_w * 0.6),
                             QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
        p.setPen(pointer)
        p.drawLine(
            QtCore.QPointF(cx + r_inner * math.cos(angle), cy - r_inner * math.sin(angle)),
            QtCore.QPointF(cx + r_outer * math.cos(angle), cy - r_outer * math.sin(angle)),
        )
        p.end()
