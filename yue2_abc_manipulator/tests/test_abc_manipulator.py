import importlib.util
from pathlib import Path
import sys
import unittest

# Load the yue2_abc_manipulator submodule in isolation
SUBMOD = Path(__file__).resolve().parents[1]

pkg = type(sys)('yue2_abc_manipulator')
pkg.__path__ = [str(SUBMOD)]
sys.modules['yue2_abc_manipulator'] = pkg

# Load abc_mute
abc_mute_spec = importlib.util.spec_from_file_location(
    'yue2_abc_manipulator.abc_mute', SUBMOD / 'abc_mute.py'
)
abc_mute = importlib.util.module_from_spec(abc_mute_spec)
sys.modules['yue2_abc_manipulator.abc_mute'] = abc_mute
abc_mute_spec.loader.exec_module(abc_mute)

# Load instrumentalize
inst_spec = importlib.util.spec_from_file_location(
    'yue2_abc_manipulator.instrumentalize', SUBMOD / 'instrumentalize.py'
)
inst_mod = importlib.util.module_from_spec(inst_spec)
sys.modules['yue2_abc_manipulator.instrumentalize'] = inst_mod
inst_spec.loader.exec_module(inst_mod)

# Load abc_ops
abc_ops_spec = importlib.util.spec_from_file_location(
    'yue2_abc_manipulator.abc_ops', SUBMOD / 'abc_ops.py'
)
abc_ops = importlib.util.module_from_spec(abc_ops_spec)
sys.modules['yue2_abc_manipulator.abc_ops'] = abc_ops
abc_ops_spec.loader.exec_module(abc_ops)

# Load nodes
nodes_spec = importlib.util.spec_from_file_location(
    'yue2_abc_manipulator.nodes', SUBMOD / 'nodes.py'
)
nodes_mod = importlib.util.module_from_spec(nodes_spec)
sys.modules['yue2_abc_manipulator.nodes'] = nodes_mod
nodes_spec.loader.exec_module(nodes_mod)

manipulate = nodes_mod.NODE_CLASS_MAPPINGS['YuE2ABCManipulator']().process
transpose_abc = abc_ops.transpose_abc
change_bpm = abc_ops.change_bpm
change_meter = abc_ops.change_meter


SAMPLE = """X:1
T:
M:4/4
L:1/16
Q:1/4=90
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:G
V: Vocal
Z2|
V: Ins
G,8D,4G,4|B,4A,4B,4C4|
"""


class TransposeTests(unittest.TestCase):
    def test_zero_is_identity(self):
        self.assertEqual(transpose_abc(SAMPLE, 0), SAMPLE)

    def test_up_one_octave(self):
        result = transpose_abc('V: Vocal\nC4D4|\n', 12)
        self.assertIn('c4d4|', result)

    def test_down_one_octave(self):
        result = transpose_abc('V: Vocal\nc4d4|\n', -12)
        self.assertIn('C4D4|', result)

    def test_up_semitone(self):
        result = transpose_abc('V: Vocal\nC4|\n', 1)
        self.assertIn('^C4|', result)

    def test_up_two_semitones(self):
        result = transpose_abc('V: Vocal\nC4|\n', 2)
        self.assertIn('D4|', result)

    def test_down_semitone(self):
        result = transpose_abc('V: Vocal\nD4|\n', -1)
        self.assertIn('^C4|', result)

    def test_preserves_rests(self):
        result = transpose_abc('V: Vocal\nz4Z2|\n', 5)
        self.assertIn('z4Z2|', result)

    def test_preserves_quotes_and_comments(self):
        result = transpose_abc('V: Vocal\n"Cmaj7"C4| % comment\n', 2)
        self.assertIn('"Cmaj7"', result)
        self.assertIn('% comment', result)

    def test_preserves_ties(self):
        result = transpose_abc('V: Vocal\nC4-|\n', 2)
        self.assertIn('D4-|', result)

    def test_octave_wrap_up(self):
        result = transpose_abc('V: Vocal\nB4|\n', 1)
        self.assertIn('c4|', result)

    def test_octave_wrap_down(self):
        result = transpose_abc('V: Vocal\nc4|\n', -1)
        self.assertIn('B4|', result)

    def test_two_octaves_up(self):
        result = transpose_abc('V: Vocal\nC4|\n', 24)
        self.assertIn("c'4|", result)

    def test_two_octaves_down(self):
        result = transpose_abc("V: Vocal\nc4|\n", -24)
        self.assertIn('C,4|', result)

    def test_accidentals_preserved_through_transpose(self):
        # ^F (= F#) up 2 semitones -> ^G (= G#)
        result = transpose_abc('V: Vocal\n^F4|\n', 2)
        self.assertIn('^G4|', result)

    def test_inline_key_untouched(self):
        result = transpose_abc('V: Vocal\nC4[K:G]D4|\n', 2)
        self.assertIn('[K:G]', result)


class BPMTests(unittest.TestCase):
    def test_change_bpm(self):
        result = change_bpm(SAMPLE, 120)
        self.assertIn('Q:1/4=120', result)
        self.assertNotIn('Q:1/4=90', result)

    def test_zero_bpm_no_change(self):
        result = change_bpm(SAMPLE, 0)
        self.assertIn('Q:1/4=90', result)

    def test_only_first_q_changed(self):
        abc = 'Q:1/4=90\nQ:1/4=100\n'
        result = change_bpm(abc, 120)
        self.assertIn('Q:1/4=120', result)
        self.assertIn('Q:1/4=100', result)


class MeterTests(unittest.TestCase):
    def test_change_meter(self):
        result = change_meter(SAMPLE, '3/4')
        self.assertIn('M:3/4', result)
        self.assertNotIn('M:4/4', result)

    def test_empty_meter_no_change(self):
        result = change_meter(SAMPLE, '')
        self.assertIn('M:4/4', result)


class NodeIntegrationTests(unittest.TestCase):
    def test_bypass(self):
        abc, report = manipulate(SAMPLE, 'bypass', 0, 'unchanged', 0)
        self.assertEqual(abc, SAMPLE)
        self.assertIn('bypass', report)

    def test_combined_bpm_and_transpose(self):
        abc, report = manipulate(SAMPLE, 'bypass', 120, 'unchanged', 2)
        self.assertIn('Q:1/4=120', abc)
        self.assertIn('BPM set to 120', report)
        self.assertIn('Transposed up 2 semitones', report)

    def test_vocalize_mode(self):
        abc, report = manipulate(SAMPLE, 'vocalize', 0, 'unchanged', 0)
        self.assertIn('Moved', report)

    def test_instrumentalize_mode(self):
        src = """M:4/4
L:1/8
K:C
V: Vocal
C2D2E2F2|
V: Ins
Z|
"""
        abc, report = manipulate(src, 'instrumentalize', 0, 'unchanged', 0)
        self.assertIn('Moved', report)

    def test_all_operations(self):
        abc, report = manipulate(SAMPLE, 'vocalize', 100, '3/4', -3)
        self.assertIn('Moved', report)
        self.assertIn('BPM set to 100', report)
        self.assertIn('Time signature set to 3/4', report)
        self.assertIn('Transposed down 3 semitones', report)

    def test_return_types(self):
        node = nodes_mod.NODE_CLASS_MAPPINGS['YuE2ABCManipulator']()
        self.assertEqual(node.RETURN_TYPES, ('STRING', 'STRING'))
        self.assertEqual(node.RETURN_NAMES, ('abc', 'report'))
        self.assertEqual(node.CATEGORY, 'YuE2/ABC')

    def test_registration(self):
        self.assertIn('YuE2ABCManipulator', nodes_mod.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            nodes_mod.NODE_DISPLAY_NAME_MAPPINGS['YuE2ABCManipulator'],
            'YuE2 ABC Manipulator'
        )


if __name__ == '__main__':
    unittest.main()
