# PathSnap model ownership

Paths in the table are relative to `src/pathsnap/` unless marked as repository directories.

| Owner | Model files | Other owned work |
| --- | --- | --- |
| Krishna Manoj | `logistic_regression.py` | `candidates.py`, `decoder.py`, candidate miss rate and decoder comparison |
| Krishna H Zalavadiya | `random_forest.py` | Shared `training.py`, `evaluation.py`, and `tuning.py` |
| Krishna Prajoth Grandhi | `gaussian_naive_bayes.py` | `label_alignment.py`, `inference.py`, and the repository's `api/` directory |
| Krithika S | `decision_tree.py`, `hist_gradient_boosting.py` | `visualization.py` and the repository's `ui/` directory |

Use a branch for your work and coordinate changes to shared modules with their owner.

## Proposed model interface

The logistic regression module implements this interface. Confirm it with Krishna H before wiring the shared runner.

- The training function accepts `(trip_id, Candidate)` rows and a label mapping from `(trip_id, observation_index)` to the true directed arc ID. Supply the training subset for each fold. Keep every observation from the same trip in the same fold.
- `score_candidates(model, candidates)` takes the fitted model and a sequence of candidate rows. It returns the probability that each candidate is the true arc, in the same order as the input rows.
- The shared evaluator can use those probabilities to select the highest-scoring candidate per fix. For Viterbi, pass the decoder a sequence of fixes, each containing `(arc_id, probability)` pairs. Preserve empty fixes so coverage and decoder outputs stay aligned to source observation indices.
