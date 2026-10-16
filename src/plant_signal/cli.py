"""Task registry and subprocess launcher; listing tasks needs no ML or UI imports."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from plant_signal.paths import ROOT_DIR


@dataclass(frozen=True)
class Task:
    module: str
    description: str
    web: bool = False


TASKS = {
    'annotate': Task('plant_signal.ui.annotation', 'Acquire and annotate trials', True),
    'mcu': Task('plant_signal.ui.mcu', 'Display MCU classifications', True),
    'monitor': Task('plant_signal.ui.realtime', 'PC realtime inference', True),
    'monitor-v': Task('plant_signal.ui.realtime_v', 'PC inference with V ... uV input', True),
    'period': Task('plant_signal.ui.period', 'Periodic noise measurement', True),
    'inspect': Task('plant_signal.training.inspect', 'Inspect labeled data'),
    'export-labels': Task('plant_signal.annotation.export', 'Update labeled trial exports'),
    'build-tags': Task('plant_signal.annotation.windows', 'Build legacy windows from trial tags'),
    'build-annotations': Task('plant_signal.training.legacy.build_annotations', 'Build windows from legacy CSV annotations'),
    'train-binary': Task('plant_signal.training.binary', 'Train binary window classifier'),
    'train-event': Task('plant_signal.training.event', 'Train Impact event detector'),
    'experiments': Task('plant_signal.training.experiments', 'Compare binary training candidates'),
    'train-legacy': Task('plant_signal.training.legacy.pipeline', 'Run full legacy three-class pipeline'),
    'train': Task('plant_signal.training.legacy.train', 'Train legacy three-class model'),
    'evaluate': Task('plant_signal.training.legacy.evaluate', 'Evaluate legacy Keras model'),
    'quantize': Task('plant_signal.training.legacy.quantize', 'Quantize legacy model to INT8'),
    'test-tflite': Task('plant_signal.training.legacy.tflite_test', 'Evaluate legacy INT8 model'),
    'export-c': Task('plant_signal.training.legacy.export_c', 'Export legacy TFLite C arrays'),
    'demo': Task('plant_signal.training.legacy.demo', 'Generate synthetic demo data'),
    'report': Task('plant_signal.reporting.materials', 'Generate report figures'),
}


def build_command(task_name: str, arguments: list[str], python: str | None = None) -> list[str]:
    """Build an argument list; never use a shell command string."""
    task = TASKS[task_name]
    command = [python or sys.executable, '-B']
    if task.web:
        spec = importlib.util.find_spec(task.module)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(task.module)
        command += ['-m', 'streamlit', 'run', spec.origin]
    else:
        command += ['-m', task.module]
    return command + list(arguments)


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = source_root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Plant signal acquisition, training and monitoring tasks.')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without starting it; place before TASK.')
    parser.add_argument('task', nargs='?', default='list', choices=['list', *TASKS])
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Arguments passed to the selected task.')
    args = parser.parse_args(argv)
    if args.task == 'list':
        for name, task in TASKS.items():
            print(f'{name:20} {task.description}')
        return 0
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    command = build_command(args.task, forwarded)
    if args.dry_run:
        print(json.dumps({'task': args.task, 'cwd': str(ROOT_DIR), 'command': command}, ensure_ascii=False, indent=2))
        return 0
    try:
        return subprocess.run(command, cwd=ROOT_DIR, env=child_environment(), check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
