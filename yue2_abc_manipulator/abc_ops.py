"""YuE2 ABC text manipulations: instrumentalize, vocalize, transpose, BPM, meter."""

import re

FIELD = re.compile(r"^\s*([A-Za-z]):(.*)")
INLINE = re.compile(r"\[([A-Za-z]):([^\]]*)\]")
NOTE = re.compile(
    r"(?P<acc>\^{1,2}|_{1,2}|=)?(?P<pitch>[A-Ga-g])(?P<oct>[,']*)(?P<len>\d+(?:/+\d*)?|/+\d*)?(?P<tie>-?)"
)

_CHROMATIC = [
    ('C', ''), ('C', '^'), ('D', ''), ('D', '^'), ('E', ''),
    ('F', ''), ('F', '^'), ('G', ''), ('G', '^'), ('A', ''), ('A', '^'), ('B', ''),
]


def _semitone_index(letter, acc):
    """Map a note letter + accidental to a semitone index 0-11."""
    base = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[letter.upper()]
    if acc in ('^', '^^'):
        base += len(acc)
    elif acc in ('_', '__'):
        base -= len(acc)
    return base % 12


def _transpose_note(pitch_char, octave_str, acc, semitones):
    """Transpose a note by the given number of semitones.

    Returns (new_pitch_char, new_octave_str, new_acc_str).
    """
    is_lower = pitch_char.islower()
    letter = pitch_char.upper()
    # Count absolute octave: C, = octave 0; C = octave 1; c = octave 2; c' = octave 3 ...
    if is_lower:
        octave = 2
        for ch in octave_str:
            if ch == "'":
                octave += 1
            elif ch == ",":
                octave -= 1
    else:
        octave = 1
        for ch in octave_str:
            if ch == ",":
                octave -= 1
            elif ch == "'":
                octave += 1

    semi = _semitone_index(letter, acc or '') + semitones
    # Compute new octave offset from transposition
    octave += semi // 12
    semi = semi % 12
    # Find nearest match in chromatic table
    new_letter, new_acc_raw = _CHROMATIC[semi]
    # Determine ABC case and octave marks
    if octave >= 2:
        new_pitch = new_letter.lower()
        new_oct = "'" * (octave - 2)
    elif octave == 1:
        new_pitch = new_letter.upper()
        new_oct = ''
    else:
        new_pitch = new_letter.upper()
        new_oct = ',' * (1 - octave)
    # Build accidental string — only emit if needed
    new_acc = new_acc_raw  # '' or '^'
    return new_pitch, new_oct, new_acc


def _voice_id(value):
    parts = value.strip().split()
    return parts[0] if parts else None


def transpose_abc(abc, semitones):
    """Transpose all notes in an ABC string by the given number of semitones."""
    if semitones == 0:
        return abc
    output = []
    active_voice = None
    for line in abc.splitlines(keepends=True):
        field = FIELD.match(line)
        if field:
            if field[1] == "V":
                active_voice = _voice_id(field[2])
            output.append(line)
            continue
        # Process music line
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            if char == '%':
                result.append(line[i:])
                break
            if char in ('"', '!', '+'):
                end = i + 1
                while end < len(line):
                    if line[end] == '\\':
                        end += 2
                        continue
                    if line[end] == char:
                        break
                    end += 1
                if end >= len(line):
                    result.append(line[i:])
                    break
                result.append(line[i:end + 1])
                i = end + 1
                continue
            inline = INLINE.match(line, i)
            if inline:
                if inline[1] == "V":
                    active_voice = _voice_id(inline[2])
                result.append(inline[0])
                i = inline.end()
                continue
            note = NOTE.match(line, i)
            if note and note['pitch']:
                acc = note['acc'] or ''
                pitch_char = note['pitch']
                octave_str = note['oct'] or ''
                length = note['len'] or ''
                tie = note['tie'] or ''
                new_pitch, new_oct, new_acc = _transpose_note(pitch_char, octave_str, acc, semitones)
                result.append(new_acc + new_pitch + new_oct + length + tie)
                i = note.end()
                continue
            result.append(char)
            i += 1
        output.append(''.join(result))
    return ''.join(output)


def change_bpm(abc, new_bpm):
    """Replace the Q: tempo field value."""
    if new_bpm <= 0:
        return abc
    def replace_q(m):
        prefix = m.group(1)
        return f'{prefix}1/4={new_bpm}'
    return re.sub(r'^(\s*Q:)\s*.*$', replace_q, abc, count=1, flags=re.MULTILINE)


def change_meter(abc, new_meter):
    """Replace all M: fields with a new time signature like '3/4'."""
    if not new_meter:
        return abc
    def replace_m(m):
        prefix = m.group(1)
        return f'{prefix}{new_meter}'
    return re.sub(r'^(\s*M:)\s*.*$', replace_m, abc, flags=re.MULTILINE)
