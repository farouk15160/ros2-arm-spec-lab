"""Portable benchmark JSON, Markdown and comparison/error plots."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _atomic_text(path, contents):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_benchmark_report(result, directory, simulation=None, reference=None):
    """Write machine-readable results, Markdown and plots; return artifact paths.

    Plotting uses aligned data embedded by compare_runs. simulation/reference
    arguments are accepted for producer convenience; no different data are used
    to re-compute or silently replace the supplied result.
    """
    contents = json.dumps(result, indent=2, allow_nan=False) + '\n'
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {'json': folder / 'benchmark.json', 'markdown': folder / 'benchmark.md'}
    plots, plot_note = _plots(result, folder)
    _atomic_text(paths['json'], contents)
    _atomic_text(paths['markdown'], _markdown(result, plots, plot_note))
    return {**paths, 'plots': plots}


def _markdown(result, plots, plot_note):
    lines = [f"# {result['robot']} Benchmark", '', f"Trajectory/scenario: `{result['scenario_id']}`",
             '', f"Result: **{result['status']}**", '',
             'Only explicitly configured tolerances determine acceptance. Missing requested data cannot pass.', '']
    for role, evidence in result['evidence'].items():
        lines.extend([f"{role.capitalize()} evidence: **{evidence['kind']}** — {evidence['source']}", ''])
    lines += ['| Channel | RMSE | Maximum error | Final error | Unit | Tolerance | Status |',
              '|---|---:|---:|---:|---|---:|---|']
    for name, metric in result['metrics'].items():
        values = [metric.get(key, '—') for key in ('rmse', 'max', 'final', 'unit', 'tolerance', 'status')]
        lines.append('| ' + name + ' | ' + ' | '.join(_format(value) for value in values) + ' |')
        for joint, component in metric.get('per_joint', {}).items():
            values = [component.get(key, '—') for key in ('rmse', 'max', 'final', 'unit', 'tolerance', 'status')]
            lines.append('| ' + name + '/' + joint + ' | ' + ' | '.join(_format(value) for value in values) + ' |')
    lines += ['', f"Alignment: `{result['alignment']}`", '']
    for name in ('duration', 'energy', 'effort'):
        if name in result:
            lines += [f"{name.capitalize()}: `{result[name]}`", '']
    for channel in ('tcp_position', 'tcp_orientation', 'joint_torque'):
        if channel in result['series']:
            title = 'TCP final ' + channel.removeprefix('tcp_') if channel.startswith('tcp_') else 'Final joint torques'
            lines += [title + ' (end of overlap):', '']
            for role in ('simulation', 'reference'):
                lines += [f"- {role}: `{result['series'][channel][role][-1]}`"]
            lines.append('')
    for path in plots:
        lines += [f'![{path.stem}]({path.name})', '']
    if plot_note:
        lines += [plot_note, '']
    lines += ['Limitations:', ''] + ['- ' + item for item in result['limitations']]
    return '\n'.join(lines) + '\n'


def _format(value):
    if value is None:
        return '—'
    return f'{value:.8g}' if isinstance(value, (int, float)) else str(value)


def _plots(result, folder):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        return [], 'Plots unavailable: install matplotlib. Numerical results remain available.'
    series = result['series']
    times = series['time']
    paths = []
    for channel, values in series.items():
        if channel == 'time':
            continue
        fig = Figure(figsize=(9, 4))
        FigureCanvasAgg(fig)
        axis = fig.subplots()
        for role, style in (('simulation', '-'), ('reference', '--')):
            values_array = np.asarray(values[role])
            for index in range(values_array.shape[1]):
                label = result['joint_names'][index] if channel.startswith('joint_') else 'xyzw'[index]
                axis.plot(times, values_array[:, index], style, label=f'{role} {label}')
        unit = 'quaternion xyzw' if channel == 'tcp_orientation' else result['metrics'][channel]['unit']
        axis.set(xlabel='Time [s]', ylabel=unit, title=channel)
        axis.legend(fontsize='x-small', ncol=2)
        fig.tight_layout()
        path = folder / (channel + '.png')
        fig.savefig(path)
        paths.append(path)
    paths.extend(_error_plot(result, folder, Figure, FigureCanvasAgg))
    if 'tcp_position' in series:
        fig = Figure(figsize=(12, 4))
        FigureCanvasAgg(fig)
        for axis, (i, j) in zip(fig.subplots(1, 3), ((0, 1), (0, 2), (1, 2))):
            for role in ('simulation', 'reference'):
                xyz = np.asarray(series['tcp_position'][role])
                axis.plot(xyz[:, i], xyz[:, j], label=role)
            axis.set(xlabel='xyz'[i] + ' [m]', ylabel='xyz'[j] + ' [m]', title='TCP projection')
            axis.legend()
        fig.tight_layout()
        path = folder / 'tcp_trajectory.png'
        fig.savefig(path)
        paths.append(path)
    return paths, None


def _error_plot(result, folder, figure_type, canvas_type):
    channels = [key for key in result['series'] if key != 'time']
    if not channels:
        return []
    fig = figure_type(figsize=(9, max(3, len(channels) * 2.4)))
    canvas_type(fig)
    axes = fig.subplots(len(channels), 1, squeeze=False)
    for axis, channel in zip(axes[:, 0], channels):
        errors = np.asarray(result['series'][channel]['error'])
        axis.plot(result['series']['time'], errors)
        axis.set(xlabel='Time [s]', ylabel=result['metrics'][channel]['unit'], title=channel + ' error')
    fig.tight_layout()
    path = folder / 'errors.png'
    fig.savefig(path)
    return [path]
