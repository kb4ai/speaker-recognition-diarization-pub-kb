# RTTM Format Specification

## Overview

**Rich Transcription Time Marked (RTTM)** is the standard annotation format for speaker diarization. It originated from NIST's Rich Transcription evaluations.

## Format Structure

RTTM is a space-delimited text file with one segment per line:

```
TYPE FILE CHANNEL START DURATION <NA> <NA> SPEAKER <NA> <NA>
```

### Field Definitions

| Position | Field | Description | Example |
|----------|-------|-------------|---------|
| 1 | TYPE | Segment type | SPEAKER |
| 2 | FILE | File identifier (no extension) | meeting_001 |
| 3 | CHANNEL | Audio channel (1-indexed) | 1 |
| 4 | START | Start time in seconds | 10.250 |
| 5 | DURATION | Duration in seconds | 5.300 |
| 6 | ORTHO | Orthographic transcription | `<NA>` |
| 7 | STYPE | Speaker type | `<NA>` |
| 8 | SPEAKER | Speaker identifier | SPEAKER_01 |
| 9 | CONF | Confidence score | `<NA>` |
| 10 | SLAT | Signal lookahead time | `<NA>` |

## Example RTTM File

```
SPEAKER meeting_001 1 0.500 2.350 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER meeting_001 1 3.200 4.100 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER meeting_001 1 7.500 1.800 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER meeting_001 1 8.900 3.200 <NA> <NA> SPEAKER_02 <NA> <NA>
SPEAKER meeting_001 1 10.100 2.500 <NA> <NA> SPEAKER_01 <NA> <NA>
```

## Key Rules

1. **Fields are space-separated**: Use exactly one space between fields
2. **Time is in seconds**: Floating-point with millisecond precision
3. **`<NA>` for unused fields**: Required placeholder for optional fields
4. **One segment per line**: Each speaking turn is a separate line
5. **File ID excludes extension**: Use `audio_001` not `audio_001.wav`
6. **Channel is 1-indexed**: First channel is 1, not 0

## Handling Overlapping Speech

RTTM naturally represents overlapping speech with segments that share time intervals:

```
SPEAKER call_123 1 5.000 3.000 <NA> <NA> SPEAKER_A <NA> <NA>
SPEAKER call_123 1 6.500 2.500 <NA> <NA> SPEAKER_B <NA> <NA>
```

In this example, SPEAKER_A and SPEAKER_B overlap from 6.5s to 8.0s.

## Python Examples

### Writing RTTM

```python
def write_rttm(segments, file_id, output_path):
    """
    Write segments to RTTM file.

    segments: list of (start, end, speaker_id) tuples
    """
    with open(output_path, 'w') as f:
        for start, end, speaker in segments:
            duration = end - start
            line = f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            f.write(line)
```

### Reading RTTM

```python
def read_rttm(rttm_path):
    """
    Read RTTM file and return segments.

    Returns: list of (file_id, start, end, speaker_id) tuples
    """
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                file_id = parts[1]
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append((file_id, start, start + duration, speaker))
    return segments
```

### Using Pyannote

```python
from pyannote.core import Annotation, Segment

# Write RTTM from pyannote Annotation
annotation = pipeline("audio.wav")
with open("output.rttm", "w") as f:
    annotation.write_rttm(f)

# Read RTTM to pyannote Annotation
from pyannote.database.util import load_rttm
annotations = load_rttm("reference.rttm")
```

## Validation Checklist

* [ ] All fields present (10 per line)
* [ ] TYPE is "SPEAKER" for diarization
* [ ] Times are non-negative floats
* [ ] Duration is positive
* [ ] No overlapping segments for same speaker (redundant)
* [ ] File IDs are consistent
* [ ] Speaker IDs follow naming convention

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Extra spaces | Multiple spaces between fields | Use single spaces |
| Missing fields | Incomplete lines | Add `<NA>` placeholders |
| Negative times | Incorrect calculation | Check start time logic |
| Zero duration | Segment too short | Merge with adjacent |

## Related Formats

* **CTM**: Time-marked word-level transcriptions
* **STM**: Segment time-marked transcriptions
* **UEM**: Unpartitioned evaluation map (scoring regions)

## Tools

* **dscore**: NIST scoring toolkit
* **pyannote.database**: RTTM I/O utilities
* **rttm-viewer**: Visual inspection tool
* **md-eval.pl**: Traditional Perl evaluation script

## References

* NIST Rich Transcription Meeting Recognition Evaluation Plan
* DIHARD Challenge Data Format Specification
* Pyannote Database Documentation
