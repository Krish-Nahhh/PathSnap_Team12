# Viterbi decoder handoff

After classification, use the GPS timestamps to restart Viterbi at gaps:

```python
from src.pathsnap.decoder import decode_route
from src.pathsnap.label_alignment import detect_gps_gaps

segment_starts = [gap.end_index for gap in detect_gps_gaps(track)]
predictions = decode_route(observations, arcs, segment_starts=segment_starts)
```

`track` contains (longitude, latitude, timestamp in seconds) rows. The
shared detector flags gaps over five seconds and non-increasing
timestamps. Viterbi restarts before the later fix, preserving both fixes.

- Pass one list of `(arc_id, positive_class_probability)` pairs per GPS
  fix in source order. Keep empty lists for fixes without candidates.
- Pass the graph's directed `(from_node, to_node)` pairs in source file
  order. Arc IDs are the original zero-based rows in that list.
- The return value has one arc ID or `None` per input fix. An empty fix
  ends the current decoded segment. The output represents per-fix arc
  assignments. A route renderer must expand any skipped connector arcs
  separately if it needs a complete graph path.
- The decoder uses log classifier probabilities and graph transition
  penalties. Defaults allow up to 3 directed connector arcs, with a
  penalty of 1 per connector and 8 when no bounded connection exists.
  A disconnected transition is possible when candidate coverage requires it.

The decoder accepts scores from any selected classifier. The comparison
runner uses the existing logistic feature adapter. To evaluate a
classifier with a different feature schema, use its scoring adapter in
`predict_trip` and keep both methods on the same scores.

## Reproduce the student-test comparison

Run from the team checkout with the model's Python dependencies available.

```powershell
$env:PYTHONPATH = (Resolve-Path '.venv/pathsnap-deps').Path
python run_decoder_comparison.py 'Student folder/Student folder/student_test' results/logistic_regression/logistic_regression.joblib results/decoder_comparison.csv
```

The logistic artifact was fitted on training data. The comparison constructs
predictions from the GPS track and directed road graph. Reference routes
are used for scoring after prediction. Trip 00000000 was skipped because
its source directory lacks the .arcs file.

| Metric | Classifier only | Viterbi |
| --- | ---: | ---: |
| Per-fix accuracy | 73.76% | 77.37% |
| Mean trip sequence similarity | 0.591 | 0.707 |
| Direct continuity | 87.04% | 99.10% |

Accuracy pools 16,657 fixes across nine trips and uses the existing
reference-route alignment. Sequence similarity averages each trip's
SequenceMatcher ratio after collapsing consecutive repetitions in the
prediction and reference. Direct continuity pools adjacent scored fixes
whose arcs are the same or touch in the forward direction. This metric
counts transitions that skip connector arcs as indirect. The CSV reports
per-trip metrics and decode time.

The timestamp rule flagged 44 breaks across the nine trips. Compared with
Viterbi without timestamp breaks, per-fix accuracy changed from 77.45%
to 77.37% and direct continuity from 99.18% to 99.10%. Continuity counts
all adjacent pairs with predictions, including across timestamp breaks.
Trip 00000091 had no timestamp breaks and still reached only 25.86%
per-fix accuracy with Viterbi. The continuity gain does not establish
that the selected arcs match the reference.
