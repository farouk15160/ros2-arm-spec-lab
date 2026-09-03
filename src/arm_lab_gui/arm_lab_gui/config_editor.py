"""Edit every physical parameter of the arm, then launch it in Gazebo.

    ros2 run arm_lab_gui config_editor
    ros2 run arm_lab_gui config_editor --config my_variant.yaml

The form is generated from `schema.py`, so it always covers the whole
configuration. Every edit is validated and the derived figures -- mass, reach,
payload, droop, per-joint torque -- are recomputed immediately, so the cost of a
change is visible before anything is launched.
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml
from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QFont
from python_qt_binding.QtWidgets import (QAbstractItemView, QApplication,
                                         QCheckBox, QComboBox, QDialog,
                                         QDoubleSpinBox, QFileDialog,
                                         QFormLayout, QGroupBox, QHBoxLayout,
                                         QInputDialog, QLabel, QLineEdit,
                                         QListWidget, QMainWindow, QMessageBox,
                                         QPlainTextEdit, QPushButton,
                                         QScrollArea, QSpinBox, QSplitter,
                                         QTabWidget, QTableWidget,
                                         QTableWidgetItem, QVBoxLayout, QWidget)

from arm_lab_model.config import default_config_path, load_config
from arm_lab_model.kinematics import ArmModel
from arm_lab_model.spec_report import SpecReport

from . import schema
from .widgets import ACCENT, BAD, BG, DIM, GOOD, GRID, PANEL, TEXT, WARN

STYLE = f"""
QMainWindow, QWidget {{ background: rgb({BG.red()},{BG.green()},{BG.blue()});
                        color: rgb({TEXT.red()},{TEXT.green()},{TEXT.blue()}); }}
QGroupBox {{ border: 1px solid rgb({GRID.red()},{GRID.green()},{GRID.blue()});
             border-radius: 3px; margin-top: 10px; padding-top: 10px;
             font-weight: bold; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px;
                    color: rgb({DIM.red()},{DIM.green()},{DIM.blue()}); }}
QPushButton {{ background: rgb(52,58,68); border: 1px solid rgb(72,80,92);
               border-radius: 3px; padding: 6px 12px; }}
QPushButton:hover {{ background: rgb(64,72,86); }}
QPushButton#primary {{ background: rgb(38,92,150); border-color: rgb(64,132,200); }}
QPushButton#danger {{ background: rgb(120,44,40); border-color: rgb(170,70,64); }}
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit, QListWidget, QTableWidget
    {{ background: rgb(44,49,58); border: 1px solid rgb(72,80,92);
       border-radius: 3px; padding: 3px; selection-background-color: rgb(52,96,144); }}
QTabWidget::pane {{ border: 1px solid rgb({GRID.red()},{GRID.green()},{GRID.blue()}); }}
QTabBar::tab {{ background: rgb(44,49,58); padding: 7px 14px; margin-right: 2px;
                border-top-left-radius: 3px; border-top-right-radius: 3px; }}
QTabBar::tab:selected {{ background: rgb(62,70,84); }}
QHeaderView::section {{ background: rgb(52,58,68); padding: 4px; border: 0; }}
"""


# --------------------------------------------------------------- dict paths
def get_path(data: Any, path: str, default=None) -> Any:
    node = data
    for key in path.split('.'):
        try:
            node = node[int(key)] if isinstance(node, list) else node[key]
        except (KeyError, IndexError, ValueError, TypeError):
            return default
    return node


def set_path(data: Any, path: str, value: Any) -> None:
    keys = path.split('.')
    node = data
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node.setdefault(key, {})
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


class Vec3Widget(QWidget):
    """Three spin boxes that behave as one value."""

    def __init__(self, on_change: Callable[[], None], decimals: int = 4):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.boxes = []
        for _ in range(3):
            box = QDoubleSpinBox()
            box.setRange(-1e6, 1e6)
            box.setDecimals(decimals)
            box.setSingleStep(0.01)
            box.valueChanged.connect(lambda _v: on_change())
            row.addWidget(box)
            self.boxes.append(box)

    def value(self) -> List[float]:
        return [b.value() for b in self.boxes]

    def set_value(self, values) -> None:
        for box, v in zip(self.boxes, list(values) + [0.0, 0.0, 0.0]):
            box.blockSignals(True)
            box.setValue(float(v))
            box.blockSignals(False)


class ConfigEditor(QMainWindow):
    def __init__(self, path: Optional[str] = None):
        super().__init__()
        self.path = path or default_config_path()
        with open(self.path) as fh:
            self.data: Dict[str, Any] = yaml.safe_load(fh)
        self.original = copy.deepcopy(self.data)
        self.widgets: Dict[str, Any] = {}
        self.joint_index = 0
        self._loading = False
        self._process: Optional[subprocess.Popen] = None

        self.setWindowTitle('Arm configuration')
        self.setStyleSheet(STYLE)
        self.resize(1420, 940)
        self._build()
        self._reload_widgets()
        self._recompute()

    # ------------------------------------------------------------- building
    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        self.tabs.addTab(self._scroll(self._groups_page(schema.robot_groups())),
                         'Robot')
        self.tabs.addTab(self._joints_page(), 'Links && joints')
        self.tabs.addTab(self._table_page('materials', schema.MATERIAL_FIELDS,
                                          'Material'), 'Materials')
        self.tabs.addTab(self._table_page('actuators', schema.ACTUATOR_FIELDS,
                                          'Actuator'), 'Actuators')
        self.tabs.addTab(
            self._scroll(self._groups_page(schema.end_effector_groups())),
            'End effector')
        self.tabs.addTab(self._scroll(self._groups_page(schema.control_groups())),
                         'Control && bus')
        self.tabs.addTab(self._scroll(self._groups_page(schema.error_groups())),
                         'Error sources')
        self.tabs.addTab(self._scroll(self._groups_page(schema.spec_groups())),
                         'Spec targets')

        splitter.addWidget(self._derived_panel())
        splitter.setSizes([820, 600])
        outer.addWidget(self._buttons())

    def _scroll(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        area.setFrameShape(QScrollArea.NoFrame)
        return area

    def _groups_page(self, groups: List[schema.Group]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        for group in groups:
            box = QGroupBox(group.title)
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignRight)
            for field in group.fields:
                widget = self._make_widget(field)
                self.widgets[field.path] = widget
                label = QLabel(field.label + (f'  [{field.unit}]'
                                              if field.unit else ''))
                if field.tip:
                    label.setToolTip(field.tip)
                    widget.setToolTip(field.tip)
                form.addRow(label, widget)
            layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _make_widget(self, field: schema.Field) -> QWidget:
        if field.kind == 'vec3':
            return Vec3Widget(self._on_edit, field.decimals)
        if field.kind == 'bool':
            box = QCheckBox()
            box.stateChanged.connect(lambda _s: self._on_edit())
            return box
        if field.kind == 'str':
            edit = QLineEdit()
            edit.textChanged.connect(lambda _t: self._on_edit())
            return edit
        if field.kind == 'choice':
            combo = QComboBox()
            combo.setProperty('choices', field.choices)
            combo.setProperty('options', list(field.options))
            combo.currentTextChanged.connect(lambda _t: self._on_edit())
            return combo
        if field.kind == 'int':
            box = QSpinBox()
            box.setRange(int(field.minimum), int(field.maximum))
            box.setSingleStep(max(int(field.step), 1))
            box.valueChanged.connect(lambda _v: self._on_edit())
            return box
        box = QDoubleSpinBox()
        box.setRange(field.minimum, field.maximum)
        box.setDecimals(field.decimals)
        box.setSingleStep(field.step)
        box.valueChanged.connect(lambda _v: self._on_edit())
        return box

    def _joints_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        left = QVBoxLayout()
        self.joint_list = QListWidget()
        self.joint_list.setMaximumWidth(190)
        self.joint_list.currentRowChanged.connect(self._select_joint)
        left.addWidget(QLabel('Chain, base to tool'))
        left.addWidget(self.joint_list, 1)
        row = QHBoxLayout()
        for text, slot in (('+', self._add_joint), ('-', self._remove_joint),
                           ('^', lambda: self._move_joint(-1)),
                           ('v', lambda: self._move_joint(1))):
            button = QPushButton(text)
            button.setMaximumWidth(42)
            button.clicked.connect(slot)
            row.addWidget(button)
        left.addLayout(row)
        hint = QLabel('Adding a joint changes the degrees of freedom; '
                      'the URDF, the controllers and the report all follow.')
        hint.setWordWrap(True)
        hint.setStyleSheet(f'color: rgb({DIM.red()},{DIM.green()},{DIM.blue()});'
                           ' font-size: 10px;')
        left.addWidget(hint)
        layout.addLayout(left)

        self.joint_detail = QScrollArea()
        self.joint_detail.setWidgetResizable(True)
        self.joint_detail.setFrameShape(QScrollArea.NoFrame)
        layout.addWidget(self.joint_detail, 1)
        return page

    def _rebuild_joint_detail(self) -> None:
        for path in list(self.widgets):
            if path.startswith('joints.'):
                del self.widgets[path]
        groups = schema.joint_groups(self.joint_index)
        self.joint_detail.setWidget(self._groups_page(groups))

    def _table_page(self, section: str, columns, noun: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        table = QTableWidget()
        table.setColumnCount(len(columns) + 1)
        table.setHorizontalHeaderLabels(
            [noun] + [f'{label}\n[{unit}]' if unit else label
                      for _, label, unit, _ in columns])
        table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        table.itemChanged.connect(lambda _i: self._on_table_edit(section))
        layout.addWidget(table)
        row = QHBoxLayout()
        add = QPushButton(f'Add {noun.lower()}')
        add.clicked.connect(lambda: self._add_table_row(section, columns, noun))
        row.addWidget(add)
        remove = QPushButton(f'Remove selected')
        remove.clicked.connect(lambda: self._remove_table_row(section))
        row.addWidget(remove)
        row.addStretch(1)
        layout.addLayout(row)
        setattr(self, f'{section}_table', table)
        setattr(self, f'{section}_columns', columns)
        return page

    def _derived_panel(self) -> QWidget:
        box = QGroupBox('WHAT THIS CONFIGURATION GIVES YOU')
        layout = QVBoxLayout(box)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.derived = QPlainTextEdit()
        self.derived.setReadOnly(True)
        # Fixed-width report: scroll rather than wrap, or the columns break up.
        self.derived.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = QFont('monospace')
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(9)
        self.derived.setFont(font)
        layout.addWidget(self.derived, 1)
        return box

    def _buttons(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        for text, slot, name in (
                ('Open...', self._open, ''),
                ('Save', self._save, ''),
                ('Save as...', self._save_as, ''),
                ('Revert', self._revert, ''),
                ('Full spec report', self._spec_report, ''),
                ('Verify physics', self._verify, ''),
                ('Launch in Gazebo', self._launch, 'primary')):
            button = QPushButton(text)
            if name:
                button.setObjectName(name)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
        self.path_label = QLabel(self.path)
        self.path_label.setStyleSheet(
            f'color: rgb({DIM.red()},{DIM.green()},{DIM.blue()}); font-size: 10px;')
        row.addWidget(self.path_label)
        return bar

    # -------------------------------------------------------------- loading
    def _reload_widgets(self) -> None:
        self._loading = True
        self.joint_list.clear()
        for i, joint in enumerate(self.data.get('joints', [])):
            self.joint_list.addItem(f'{i + 1}. {joint.get("name", "joint")}')
        self.joint_index = min(self.joint_index,
                               max(len(self.data.get('joints', [])) - 1, 0))
        self.joint_list.setCurrentRow(self.joint_index)
        self._rebuild_joint_detail()
        self._fill_widgets()
        self._fill_table('materials', schema.MATERIAL_FIELDS)
        self._fill_table('actuators', schema.ACTUATOR_FIELDS)
        self._loading = False

    def _fill_widgets(self) -> None:
        for path, widget in self.widgets.items():
            value = get_path(self.data, path)
            if isinstance(widget, Vec3Widget):
                widget.set_value(value or [0, 0, 0])
            elif isinstance(widget, QCheckBox):
                widget.blockSignals(True)
                widget.setChecked(bool(value))
                widget.blockSignals(False)
            elif isinstance(widget, QLineEdit):
                widget.blockSignals(True)
                widget.setText('' if value is None else str(value))
                widget.blockSignals(False)
            elif isinstance(widget, QComboBox):
                widget.blockSignals(True)
                widget.clear()
                source = widget.property('choices')
                if source:
                    widget.addItems(sorted(self.data.get(source, {})))
                else:
                    widget.addItems(widget.property('options') or [])
                if value is not None:
                    index = widget.findText(str(value))
                    if index < 0:
                        widget.addItem(str(value))
                        index = widget.findText(str(value))
                    widget.setCurrentIndex(index)
                widget.blockSignals(False)
            elif isinstance(widget, QSpinBox):
                widget.blockSignals(True)
                widget.setValue(int(round(float(value))) if value is not None else 0)
                widget.blockSignals(False)
            elif isinstance(widget, QDoubleSpinBox):
                widget.blockSignals(True)
                widget.setValue(float(value) if value is not None else 0.0)
                widget.blockSignals(False)

    def _fill_table(self, section: str, columns) -> None:
        table = getattr(self, f'{section}_table')
        entries = self.data.get(section, {})
        table.blockSignals(True)
        table.setRowCount(len(entries))
        for r, (name, values) in enumerate(sorted(entries.items())):
            table.setItem(r, 0, QTableWidgetItem(name))
            for c, (key, _label, _unit, decimals) in enumerate(columns, start=1):
                value = values.get(key, 0.0)
                table.setItem(r, c, QTableWidgetItem(f'{float(value):.{decimals}g}'))
        table.resizeColumnsToContents()
        table.blockSignals(False)

    # --------------------------------------------------------------- edits
    def _on_edit(self) -> None:
        if self._loading:
            return
        for path, widget in self.widgets.items():
            if isinstance(widget, Vec3Widget):
                set_path(self.data, path, widget.value())
            elif isinstance(widget, QCheckBox):
                set_path(self.data, path, widget.isChecked())
            elif isinstance(widget, QLineEdit):
                set_path(self.data, path, widget.text())
            elif isinstance(widget, QComboBox):
                set_path(self.data, path, widget.currentText())
            elif isinstance(widget, QSpinBox):
                set_path(self.data, path, int(widget.value()))
            elif isinstance(widget, QDoubleSpinBox):
                set_path(self.data, path, float(widget.value()))
        self._refresh_joint_names()
        self._recompute()

    def _refresh_joint_names(self) -> None:
        for i, joint in enumerate(self.data.get('joints', [])):
            item = self.joint_list.item(i)
            if item is not None:
                item.setText(f'{i + 1}. {joint.get("name", "joint")}')

    def _on_table_edit(self, section: str) -> None:
        if self._loading:
            return
        table = getattr(self, f'{section}_table')
        columns = getattr(self, f'{section}_columns')
        rebuilt: Dict[str, Any] = {}
        for r in range(table.rowCount()):
            name_item = table.item(r, 0)
            if name_item is None or not name_item.text().strip():
                continue
            name = name_item.text().strip()
            entry = dict(self.data.get(section, {}).get(name, {}))
            for c, (key, _label, _unit, _dec) in enumerate(columns, start=1):
                item = table.item(r, c)
                if item is None:
                    continue
                try:
                    entry[key] = float(item.text())
                except ValueError:
                    pass
            rebuilt[name] = entry
        if rebuilt:
            self.data[section] = rebuilt
            self._loading = True
            self._fill_widgets()            # refresh the dropdowns
            self._loading = False
            self._recompute()

    def _add_table_row(self, section: str, columns, noun: str) -> None:
        name, ok = QInputDialog.getText(self, f'New {noun.lower()}',
                                        f'{noun} name:')
        if not ok or not name.strip():
            return
        template = next(iter(self.data.get(section, {}).values()), {})
        self.data.setdefault(section, {})[name.strip()] = copy.deepcopy(template)
        self._loading = True
        self._fill_table(section, columns)
        self._fill_widgets()
        self._loading = False
        self._recompute()

    def _remove_table_row(self, section: str) -> None:
        table = getattr(self, f'{section}_table')
        row = table.currentRow()
        if row < 0:
            return
        name = table.item(row, 0).text()
        if len(self.data.get(section, {})) <= 1:
            QMessageBox.warning(self, 'Cannot remove',
                                'At least one entry has to remain.')
            return
        self.data[section].pop(name, None)
        self._loading = True
        self._fill_table(section, getattr(self, f'{section}_columns'))
        self._fill_widgets()
        self._loading = False
        self._recompute()

    def _select_joint(self, index: int) -> None:
        if index < 0 or self._loading:
            return
        self.joint_index = index
        self._loading = True
        self._rebuild_joint_detail()
        self._fill_widgets()
        self._loading = False

    def _add_joint(self) -> None:
        joints = self.data.setdefault('joints', [])
        template = copy.deepcopy(joints[-1]) if joints else {}
        n = len(joints) + 1
        template['name'] = f'joint_{n}'
        template.setdefault('link', {})['name'] = f'link_{n}'
        joints.append(template)
        self.joint_index = len(joints) - 1
        self._reload_widgets()
        self._recompute()

    def _remove_joint(self) -> None:
        joints = self.data.get('joints', [])
        if len(joints) <= 1:
            QMessageBox.warning(self, 'Cannot remove',
                                'The arm needs at least one joint.')
            return
        joints.pop(self.joint_index)
        self.joint_index = max(self.joint_index - 1, 0)
        self._reload_widgets()
        self._recompute()

    def _move_joint(self, delta: int) -> None:
        joints = self.data.get('joints', [])
        target = self.joint_index + delta
        if not (0 <= target < len(joints)):
            return
        joints[self.joint_index], joints[target] = (joints[target],
                                                    joints[self.joint_index])
        self.joint_index = target
        self._reload_widgets()
        self._recompute()

    # ------------------------------------------------------------ derived
    def _write_temp(self) -> str:
        directory = os.path.join(tempfile.gettempdir(), 'arm_lab_editor')
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'candidate.yaml')
        with open(path, 'w') as fh:
            yaml.safe_dump(self.data, fh, sort_keys=False)
        return path

    def _recompute(self) -> None:
        try:
            cfg = load_config(self._write_temp())
            model = ArmModel(cfg)
        except Exception as exc:                          # noqa: BLE001
            self.status.setText(f'<b style="color:#e2584e">Invalid: {exc}</b>')
            self.derived.setPlainText(
                'Fix the error above before launching.\n\n'
                'Common causes:\n'
                '  - wall thickness at or above the outer radius\n'
                '  - a material or actuator name that no longer exists\n'
                '  - a zero-length axis or tube direction')
            return

        try:
            q_full = model.resolve_pose('full_reach')
            q_700 = model.resolve_pose('reach_700')
            fs = model.frames(q_full)
            capacity, limiting = model.payload_capacity(q_full, fs)
            want = float(cfg.spec_targets.get('payload_at_full_reach', 2.0))
            droop = model.deflection(q_full, payload=want, fs=fs)
            tau = model.gravity_torque(q_full, payload=want, fs=fs)
            speed, _ = model.max_tcp_speed(q_full, fs)
            cap700, _ = model.payload_capacity(q_700)
        except Exception as exc:                          # noqa: BLE001
            self.status.setText(
                f'<b style="color:#e6b23e">Model builds, analysis failed: {exc}</b>')
            return

        target = cfg.spec_targets
        lines: List[str] = []

        def row(label, value, unit, ok: Optional[bool] = None, note=''):
            mark = '' if ok is None else ('  ok' if ok else '  OVER')
            lines.append(f'{label:<26}{value:>12}  {unit:<7}{mark}  {note}')

        mass = cfg.arm_mass
        mass_ok = mass <= float(target.get('arm_mass_target', 1e9))
        reach = model.reach(q_full, fs)
        r_lo = float(target.get('reach_min', 0.0))
        r_hi = float(target.get('reach_max', 1e9))

        lines.append('MASS AND GEOMETRY')
        row('arm mass incl. tool', f'{mass:.3f}', 'kg', mass_ok,
            f"target {target.get('arm_mass_target', '-')}")
        row('  structure', f'{cfg.structure_mass:.3f}', 'kg')
        row('  tool', f'{cfg.end_effector.mass:.3f}', 'kg')
        row('reach from shoulder', f'{reach * 1000:.0f}', 'mm',
            r_lo <= reach <= r_hi, f'target {r_lo * 1000:.0f}-{r_hi * 1000:.0f}')
        row('geometric max reach', f'{model.geometric_max_reach * 1000:.0f}', 'mm')
        lines.append('')

        lines.append('PAYLOAD AND LOAD')
        row('payload at full reach', f'{capacity:.2f}', 'kg',
            capacity >= want, f'limited by {cfg.joint_names[limiting]}'
            if limiting >= 0 else '')
        row('payload at 700 mm', f'{cap700:.2f}', 'kg',
            cap700 >= float(target.get('payload_at_700mm', 0.0)))
        row('TCP droop under load', f'{droop["total"] * 1000:.2f}', 'mm',
            droop['total'] <= float(target.get('raw_positioning_accuracy', 1e9)),
            f'{droop["joint_compliance"] * 1000:.2f} mm of it gearboxes')
        row('max TCP speed here', f'{speed:.2f}', 'm/s')
        lines.append('')

        lines.append('JOINT TORQUE AT FULL REACH WITH THE RATED PAYLOAD')
        lines.append(f'{"joint":<12}{"actuator":<14}{"need":>9}{"peak":>9}'
                     f'{"cont":>9}{"use":>7}')
        for i, joint in enumerate(cfg.joints):
            use = abs(tau[i]) / joint.effort_limit * 100 if joint.effort_limit else 0
            flag = ' <<' if use > 100 else ''
            lines.append(f'{joint.name:<12}{joint.actuator.name:<14}'
                         f'{abs(tau[i]):>9.1f}{joint.effort_limit:>9.1f}'
                         f'{joint.actuator.output_continuous_torque:>9.1f}'
                         f'{use:>6.0f}%{flag}')
        lines.append('')
        lines.append('LINK MASSES')
        lines.append(f'{"link":<12}{"tube":>9}{"motor+fittings":>17}{"total":>9}')
        for joint in cfg.joints:
            link = joint.link
            lines.append(f'{link.name:<12}{link.tube_mass:>9.3f}'
                         f'{link.lumped_mass:>17.3f}{link.mass:>9.3f}')

        self.derived.setPlainText('\n'.join(lines))
        worst = float(np.max(np.abs(tau) / model.torque_limits)) * 100
        colour = '#4cbe78' if worst <= 75 else ('#e6b23e' if worst <= 100
                                                else '#e2584e')
        self.status.setText(
            f'<b style="color:{colour}">Valid.</b> {cfg.dof} dof, '
            f'{mass:.2f} kg, reach {reach * 1000:.0f} mm, '
            f'worst joint at {worst:.0f} % of peak torque.')

    # --------------------------------------------------------------- actions
    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open configuration', os.path.dirname(self.path),
            'YAML (*.yaml *.yml)')
        if not path:
            return
        with open(path) as fh:
            self.data = yaml.safe_load(fh)
        self.original = copy.deepcopy(self.data)
        self.path = path
        self.path_label.setText(path)
        self._reload_widgets()
        self._recompute()

    def _save(self) -> None:
        if os.path.abspath(self.path) == os.path.abspath(default_config_path()):
            answer = QMessageBox.question(
                self, 'Overwrite the shipped configuration?',
                'This is the installed default configuration. Save a copy '
                'instead?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Yes:
                self._save_as()
                return
        self._write(self.path)

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save configuration as', os.path.expanduser('~/arm_variant.yaml'),
            'YAML (*.yaml *.yml)')
        if path:
            self._write(path)
            self.path = path
            self.path_label.setText(path)

    def _write(self, path: str) -> None:
        with open(path, 'w') as fh:
            yaml.safe_dump(self.data, fh, sort_keys=False, default_flow_style=False)
        self.original = copy.deepcopy(self.data)
        QMessageBox.information(self, 'Saved', f'Written to {path}')

    def _revert(self) -> None:
        self.data = copy.deepcopy(self.original)
        self._reload_widgets()
        self._recompute()

    def _text_dialog(self, title: str, text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(960, 780)
        dialog.setStyleSheet(STYLE)
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        font = QFont('monospace')
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(9)
        view.setFont(font)
        layout.addWidget(view)
        dialog.exec_() if hasattr(dialog, 'exec_') else dialog.exec()

    def _spec_report(self) -> None:
        try:
            cfg = load_config(self._write_temp())
            report = SpecReport(cfg)
            report.run()
            self._text_dialog('Spec report', report.render(color=False))
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, 'Spec report failed', str(exc))

    def _verify(self) -> None:
        from arm_lab_model import verification
        try:
            cfg = load_config(self._write_temp())
            checks = verification.run_all(cfg, samples=60)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, 'Verification failed', str(exc))
            return
        lines = ['Physics cross-checked against Orocos KDL, energy conservation',
                 'and closed-form results, for THIS configuration.', '']
        for c in checks:
            lines.append(f'[{"PASS" if c.passed else "FAIL"}]  {c.name}')
            lines.append(f'         {c.detail}')
        self._text_dialog('Physics verification', '\n'.join(lines))

    def _launch(self) -> None:
        path = self._write_temp()
        try:
            load_config(path)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, 'Cannot launch', f'Invalid config: {exc}')
            return
        saved = os.path.join(tempfile.gettempdir(), 'arm_lab_editor', 'launch.yaml')
        shutil.copy(path, saved)
        command = ['ros2', 'launch', 'arm_lab_bringup', 'sim.launch.py',
                   f'config_file:={saved}']
        try:
            self._process = subprocess.Popen(command)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, 'Launch failed', str(exc))
            return
        QMessageBox.information(
            self, 'Launching',
            'Gazebo, RViz and the dashboard are starting with this '
            f'configuration.\n\n{" ".join(command)}\n\n'
            'The editor stays open; edit and launch again to compare.')


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    path = None
    if '--config' in argv:
        path = argv[argv.index('--config') + 1]
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(STYLE)
    window = ConfigEditor(path)
    window.show()
    return app.exec_() if hasattr(app, 'exec_') else app.exec()


if __name__ == '__main__':
    sys.exit(main())
