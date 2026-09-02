"""Small hand-drawn Qt widgets for the capability dashboard.

Drawn with QPainter rather than pulled from a plotting library so the dashboard
has no dependency beyond python_qt_binding, which ships with ROS.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from python_qt_binding.QtCore import QPointF, QRectF, Qt
from python_qt_binding.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from python_qt_binding.QtWidgets import QSizePolicy, QWidget

BG = QColor(30, 33, 39)
PANEL = QColor(38, 42, 50)
GRID = QColor(58, 63, 73)
TEXT = QColor(222, 226, 232)
DIM = QColor(140, 148, 160)
GOOD = QColor(76, 190, 120)
WARN = QColor(230, 178, 62)
BAD = QColor(226, 88, 78)
ACCENT = QColor(84, 160, 232)


def status_color(ratio: float, warn: float = 0.75, bad: float = 1.0) -> QColor:
    """Green below `warn`, amber up to `bad`, red past it."""
    if ratio >= bad:
        return BAD
    if ratio >= warn:
        return WARN
    return GOOD


class MeterBar(QWidget):
    """A labelled bar showing a value against its limit.

    The bar fills in proportion to value/limit and turns amber then red as the
    limit approaches, so a row of them reads as a load profile at a glance.
    """

    def __init__(self, label: str = '', unit: str = '', limit: float = 1.0,
                 decimals: int = 1, parent=None):
        super().__init__(parent)
        self.label = label
        self.unit = unit
        self.limit = max(float(limit), 1e-9)
        self.decimals = decimals
        self.value = 0.0
        self.secondary: Optional[float] = None   # e.g. the simulator's own effort
        self.warn_ratio = 0.75
        self.setMinimumHeight(20)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, value: float, secondary: Optional[float] = None) -> None:
        self.value = float(value)
        self.secondary = secondary
        self.update()

    def set_limit(self, limit: float) -> None:
        self.limit = max(float(limit), 1e-9)
        self.update()

    def paintEvent(self, event):            # noqa: N802  (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect().adjusted(0, 2, -1, -2)
        p.fillRect(rect, QColor(26, 29, 34))

        ratio = abs(self.value) / self.limit
        fill = min(ratio, 1.0)
        colour = status_color(ratio, self.warn_ratio)
        bar = QRectF(rect.left(), rect.top(), rect.width() * fill, rect.height())
        p.fillRect(bar, colour)

        if ratio > 1.0:
            p.setPen(QPen(BAD, 2))
            p.drawRect(rect)

        if self.secondary is not None:
            x = rect.left() + rect.width() * min(
                abs(self.secondary) / self.limit, 1.0)
            p.setPen(QPen(QColor(255, 255, 255, 150), 1, Qt.DashLine))
            p.drawLine(int(x), rect.top(), int(x), rect.bottom())

        p.setPen(TEXT if fill < 0.55 else QColor(20, 22, 26))
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        text = f'{self.value:.{self.decimals}f} {self.unit}'.strip()
        if self.label:
            text = f'{self.label}  {text}'
        p.drawText(rect.adjusted(6, 0, -6, 0),
                   Qt.AlignVCenter | Qt.AlignLeft, text)
        p.setPen(DIM if fill < 0.9 else QColor(20, 22, 26))
        p.drawText(rect.adjusted(6, 0, -6, 0), Qt.AlignVCenter | Qt.AlignRight,
                   f'/{self.limit:.{self.decimals}f}  {ratio * 100:3.0f}%')
        p.end()


class BigNumber(QWidget):
    """One headline figure with a caption and an optional target line."""

    def __init__(self, caption: str, unit: str = '', decimals: int = 2,
                 target: Optional[float] = None, higher_is_better: bool = True,
                 parent=None):
        super().__init__(parent)
        self.caption = caption
        self.unit = unit
        self.decimals = decimals
        self.target = target
        self.higher_is_better = higher_is_better
        self.value = 0.0
        self.sub = ''
        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_value(self, value: float, sub: str = '') -> None:
        self.value = float(value)
        self.sub = sub
        self.update()

    def _colour(self) -> QColor:
        if self.target is None:
            return ACCENT
        ok = (self.value >= self.target if self.higher_is_better
              else self.value <= self.target)
        return GOOD if ok else BAD

    def paintEvent(self, event):            # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -2, -2)
        p.fillRect(rect, PANEL)
        p.setPen(QPen(GRID, 1))
        p.drawRect(rect)

        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        p.setPen(DIM)
        p.drawText(rect.adjusted(10, 6, -10, 0), Qt.AlignTop | Qt.AlignLeft,
                   self.caption)

        font.setPointSize(21)
        font.setBold(True)
        p.setFont(font)
        p.setPen(self._colour())
        value = ('inf' if self.value == float('inf')
                 else f'{self.value:.{self.decimals}f}')
        p.drawText(rect.adjusted(10, 14, -10, -14),
                   Qt.AlignVCenter | Qt.AlignLeft, f'{value} {self.unit}')

        font.setPointSize(8)
        font.setBold(False)
        p.setFont(font)
        p.setPen(DIM)
        note = self.sub
        if self.target is not None:
            arrow = '>=' if self.higher_is_better else '<='
            note = (f'{note}   ' if note else '') + \
                   f'spec {arrow} {self.target:.{self.decimals}f} {self.unit}'
        p.drawText(rect.adjusted(10, 0, -10, -6), Qt.AlignBottom | Qt.AlignLeft,
                   note)
        p.end()


class TimePlot(QWidget):
    """Rolling strip chart with an optional horizontal limit line per series."""

    def __init__(self, window_s: float = 20.0, ylabel: str = '', parent=None):
        super().__init__(parent)
        self.window_s = window_s
        self.ylabel = ylabel
        self.series: Dict[str, Dict] = {}
        self.limits: List[Tuple[float, QColor, str]] = []
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def add_series(self, name: str, colour: QColor, scale: float = 1.0) -> None:
        self.series[name] = {
            'colour': colour,
            'scale': scale,
            'points': deque(maxlen=4000),
        }

    def add_limit(self, value: float, colour: QColor = WARN, label: str = '') -> None:
        self.limits.append((value, colour, label))

    def push(self, t: float, values: Dict[str, float]) -> None:
        for name, value in values.items():
            s = self.series.get(name)
            if s is None:
                continue
            s['points'].append((t, float(value) * s['scale']))
            while s['points'] and t - s['points'][0][0] > self.window_s:
                s['points'].popleft()
        self.update()

    def clear(self) -> None:
        for s in self.series.values():
            s['points'].clear()
        self.update()

    def paintEvent(self, event):            # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -2, -2)
        p.fillRect(rect, PANEL)
        p.setPen(QPen(GRID, 1))
        p.drawRect(rect)

        plot = QRectF(rect.left() + 44, rect.top() + 8,
                      rect.width() - 54, rect.height() - 26)

        y_max = 0.0
        t_now = 0.0
        for s in self.series.values():
            for t, v in s['points']:
                y_max = max(y_max, abs(v))
                t_now = max(t_now, t)
        for value, _, _ in self.limits:
            y_max = max(y_max, value)
        y_max = y_max * 1.15 if y_max > 0 else 1.0

        font = QFont()
        font.setPointSize(7)
        p.setFont(font)
        for i in range(5):
            y = plot.bottom() - plot.height() * i / 4.0
            p.setPen(QPen(GRID, 1, Qt.DotLine))
            p.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))
            p.setPen(DIM)
            p.drawText(QRectF(rect.left() + 2, y - 8, 40, 16),
                       Qt.AlignRight | Qt.AlignVCenter, f'{y_max * i / 4.0:.2f}')

        for value, colour, label in self.limits:
            y = plot.bottom() - plot.height() * (value / y_max)
            p.setPen(QPen(colour, 1, Qt.DashLine))
            p.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))
            if label:
                p.setPen(colour)
                p.drawText(QPointF(plot.left() + 4, y - 3), label)

        t0 = t_now - self.window_s
        for name, s in self.series.items():
            pts = [pt for pt in s['points'] if pt[0] >= t0]
            if len(pts) < 2:
                continue
            poly = QPolygonF()
            for t, v in pts:
                x = plot.left() + plot.width() * (t - t0) / self.window_s
                y = plot.bottom() - plot.height() * (abs(v) / y_max)
                poly.append(QPointF(x, max(min(y, plot.bottom()), plot.top())))
            p.setPen(QPen(s['colour'], 1.6))
            p.drawPolyline(poly)

        x = plot.left() + 4
        for name, s in self.series.items():
            p.setPen(s['colour'])
            p.drawText(QPointF(x, rect.bottom() - 5), name)
            x += 9 * len(name) + 16
        if self.ylabel:
            p.setPen(DIM)
            p.drawText(QRectF(rect.left(), rect.top(), 42, 14),
                       Qt.AlignLeft | Qt.AlignVCenter, self.ylabel)
        p.end()


class ScoreCard(QWidget):
    """Live pass/fail rows against the target specification."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: List[Dict[str, object]] = []
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_rows(self, rows: List[Dict[str, object]]) -> None:
        self.rows = rows
        self.setMinimumHeight(26 + 20 * len(rows))
        self.update()

    def paintEvent(self, event):            # noqa: N802
        p = QPainter(self)
        rect = self.rect().adjusted(1, 1, -2, -2)
        p.fillRect(rect, PANEL)
        p.setPen(QPen(GRID, 1))
        p.drawRect(rect)

        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        p.setPen(DIM)
        p.drawText(rect.adjusted(10, 4, -10, 0), Qt.AlignTop | Qt.AlignLeft,
                   'LIVE vs SPEC')

        y = rect.top() + 22
        for row in self.rows:
            colour = GOOD if row['ok'] else BAD
            p.fillRect(QRectF(rect.left() + 8, y + 5, 8, 8), colour)
            p.setPen(TEXT)
            p.drawText(QRectF(rect.left() + 24, y, 130, 18),
                       Qt.AlignVCenter | Qt.AlignLeft, str(row['label']))
            p.setPen(colour)
            p.drawText(QRectF(rect.left() + 150, y, 100, 18),
                       Qt.AlignVCenter | Qt.AlignRight, str(row['text']))
            p.setPen(DIM)
            p.drawText(QRectF(rect.left() + 258, y, 120, 18),
                       Qt.AlignVCenter | Qt.AlignLeft, str(row['limit']))
            y += 20
        p.end()
