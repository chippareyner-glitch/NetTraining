import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from plant_signal.acquisition.parser import parse_serial_line
from plant_signal.annotation.windows import pair_tag_intervals
from plant_signal.cli import TASKS, build_command, main
from plant_signal.monitoring.inference import (
    dequantize_output, impact_score_from_output, infer_default_window_len,
    model_input_array, preprocess_window,
)
from plant_signal.monitoring.protocol import parse_classification_line, parse_signal_line
from plant_signal.paths import project_root


class ProtocolTests(unittest.TestCase):
    def test_acquisition_voltage_bytes_and_nulls(self):
        result = parse_serial_line(b'\x00CH0 Voltage=-12.50 uV\r\n')
        self.assertTrue(result.valid)
        self.assertEqual(result.voltage_uV, -12.5)
        self.assertEqual(result.signal_type, 'voltage')

    def test_acquisition_keeps_its_stricter_protocol(self):
        for line in ['', 'A0=12.3uA', 'V 123 uV', 'CH0 Voltage=oops uV']:
            with self.subTest(line=line):
                self.assertFalse(parse_serial_line(line).valid)

    def test_monitor_accepts_legacy_and_voltage_formats(self):
        for line in ['A0=-2.5uA', 'CH0 Voltage=-2.5 uV']:
            self.assertEqual(parse_signal_line(line), -2.5)

    def test_v_format_is_explicitly_enabled(self):
        self.assertIsNone(parse_signal_line('V      -123 uV'))
        self.assertEqual(parse_signal_line('V      -123 uV', allow_v_prefix=True), -123)
        self.assertIsNone(parse_signal_line('invalid V 123 uV', allow_v_prefix=True))

    def test_classification_is_separate_from_samples(self):
        line = 'CLS=1,LABEL=impact,CONF=127'
        self.assertEqual(parse_classification_line(line), {'class_id': 1, 'label': 'IMPACT', 'confidence': 127})
        self.assertIsNone(parse_signal_line(line, allow_v_prefix=True))
        self.assertIsNone(parse_classification_line(line + ',bad'))


class PreprocessingTests(unittest.TestCase):
    def test_event_window_uses_exported_raw_size(self):
        details = {'shape': np.array([1, 128, 1])}
        self.assertEqual(infer_default_window_len(details, {'raw_window_size': 256}), 256)
        self.assertEqual(infer_default_window_len(details, {}), 128)

    def test_downsample_baseline_unit_scale_and_clip_order(self):
        config = {'downsample_factor': 2, 'downsample_method': 'mean',
                  'baseline_points': 1, 'unit': 'mV', 'scale_mv_abs_p99_train': 0.004,
                  'clip_range': [-1, 1]}
        result = preprocess_window(np.array([0, 2, 4, 6, 8, 10]), 3, config)
        np.testing.assert_array_equal(result, np.array([0, 1, 1], dtype=np.float32))

    def test_constant_window_fallback_is_finite(self):
        result = preprocess_window(np.full(128, 1234), 128, {})
        np.testing.assert_array_equal(result, np.zeros(128))

    def test_int8_input_rounds_and_saturates(self):
        details = {'shape': [1, 4, 1], 'dtype': np.int8, 'quantization': (0.5, -1)}
        result = model_input_array(np.array([-100, 0, 1, 100]), details)
        self.assertEqual(result.shape, (1, 4, 1))
        np.testing.assert_array_equal(result.reshape(-1), [-128, -1, 1, 127])

    def test_output_dequantization_and_impact_score(self):
        output = dequantize_output(np.array([0, 64], dtype=np.int8),
                                   {'dtype': np.int8, 'quantization': (1 / 128, 0)})
        self.assertEqual(impact_score_from_output(output), 0.5)
        self.assertAlmostEqual(impact_score_from_output(np.array([0., 2.])), 0.880797, places=5)


class AnnotationTests(unittest.TestCase):
    def test_pairing_sorts_events_and_keeps_unmatched_statistics(self):
        intervals, stats = pair_tag_intervals([
            {'tag': 'impact_end', 'sample_index': 20},
            {'tag': 'impact_start', 'sample_index': 10},
            {'tag': 'noise_start', 'sample_index': 40},
        ])
        self.assertEqual([(i.kind, i.start_sample, i.end_sample) for i in intervals], [('tap', 10, 20)])
        self.assertEqual(stats['unmatched_noise_start'], 1)


class EntrypointTests(unittest.TestCase):
    def test_all_tasks_resolve_modules(self):
        for name, task in TASKS.items():
            with self.subTest(task=name):
                command = build_command(name, [], python='python-test')
                self.assertEqual(command[0], 'python-test')
                if task.web:
                    self.assertTrue(Path(command[-1]).is_file())
                else:
                    self.assertEqual(command[-1], task.module)

    def test_dry_run_preserves_arguments_with_spaces(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(['--dry-run', 'train-binary', '--out_dir', 'outputs/my experiment'])
        self.assertEqual(code, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['command'][-2:], ['--out_dir', 'outputs/my experiment'])

    def test_explicit_root_can_be_overridden_without_chdir(self):
        before = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'PLANT_SIGNAL_ROOT': tmp}):
            self.assertEqual(project_root(), Path(tmp).resolve())
        self.assertEqual(Path.cwd(), before)


if __name__ == '__main__':
    unittest.main()
