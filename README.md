# PathSnap

PathSnap is Team 12’s GPS map-matching project. It estimates which directed road arc a vehicle occupied at each timestamped GPS fix. The pipeline generates nearby candidates from the road graph, scores them with logistic regression, and uses Viterbi decoding to favor sequences that follow connected roads.

## Installation

From the project root, create a Python virtual environment and install the training dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy scikit-learn joblib
```

Place the organiser-provided development ZIP at `data/Student folder.zip`. See [Development data](data/README.md) for the input layout.

## Execution

Run these PowerShell commands from the project root. The dataset audit and candidate generator use the Python standard library.

```powershell
python inspect_dataset.py "data/Student folder.zip" --output dataset_inventory.csv
```

Extract the ZIP and set `$devRoot` to the directory containing the `train` and `student_test` folders. Export candidates for one training trip.

```powershell
$devRoot = "path/to/extracted/development-data"
python build_candidates.py generate "$devRoot/train/00000001" --output results/candidates_00000001.csv
```

The generator keeps up to 10 directed arcs within 75 metres of each fix. Arc IDs and observation indices are zero-based source row numbers. Candidate rank starts at 1. See the [sample candidate export](results/candidate_sample.csv).

If you have aligned labels for that trip, run the candidate coverage check. The labels CSV needs `split`, `trip_id`, `observation_index`, and `true_arc_id` columns. A labeled fix is a miss when its reference arc is absent from the candidate list, including fixes with zero candidates.

```powershell
python build_candidates.py score results/candidates_00000001.csv aligned_labels.csv
```

## Training and decoder experiments

Use the virtual environment from Installation for these experiments. For a saved model, use the same dependency versions that were used to train it.

The logistic regression and Viterbi tuning scripts both require the team’s `data/candidate_handoff/train_candidates.csv.gz` export and `results/aligned_labels_train.csv`. Obtain those files from the team handoff if your checkout lacks them.

```powershell
.\.venv\Scripts\python.exe run_logistic_cv.py . --output results/logistic_regression
.\.venv\Scripts\python.exe run_viterbi_tuning.py . "$devRoot/train" --output results/viterbi_tuning.json
```

The training script writes five-fold trip-grouped validation metrics and a fitted model. The tuning script fits the classifier on a subset of the training trips, searches decoder settings on held-out training trips, and reports performance on a separate set of training trips.

The student-test comparison additionally requires the team versions of `src/pathsnap/label_alignment.py` and `src/pathsnap/logistic_regression.py`, which are missing from this checkout. Once those modules are available, run the comparison with the fitted model and the extracted student-test trips.

```powershell
.\.venv\Scripts\python.exe run_decoder_comparison.py "$devRoot/student_test" results/logistic_regression/logistic_regression.joblib results/decoder_comparison.csv
```

## Deployment

The evaluator is designed for two laptops on the same local network. This checkout contains placeholders in `api/` and `ui/`. A runnable deployment requires the FastAPI and UI entry points, their dependencies, and the final inference configuration.

Once those components are integrated, deploy and verify them in this order.

1. On the backend laptop, install the inference and API dependencies. Place the fitted model, preprocessing and decoder configuration, and evaluator data on that laptop. Keep reference routes server-side for scoring after inference.
2. Start the FastAPI service bound to a LAN-accessible address and record the laptop’s LAN IP and service port.
3. On the second laptop, configure the evaluator UI with the backend’s LAN URL and start the UI.
4. Select a development trip in the UI and verify that it displays the GPS trace, predicted route, and metrics returned by the backend. Reserve evaluator-testing trips for final scoring.


## Recorded student-test results

The comparison covered 16,657 labeled fixes across nine trips. Trip `00000000` was excluded because its source directory lacked the `.arcs` file. See the [decoder notes](VITERBI.md) for metric definitions and timestamp handling.

| Metric | Classifier only | Viterbi |
| --- | ---: | ---: |
| Per-fix accuracy | 73.76% | 77.37% |
| Mean trip sequence similarity | 0.591 | 0.707 |
| Direct continuity | 87.04% | 99.10% |

## Project layout and status

| Path | Contents |
| --- | --- |
| `src/pathsnap/` | Candidate generation, model training utilities, and Viterbi decoding |
| `data/` | Local development inputs and data layout notes |
| `results/` | Candidate samples, experiment outputs, and model artifacts |
| `api/`, `ui/`, `artifacts/` | Placeholders for the evaluator service, interface, and deployment exports |
| `guidelines/`, `deliverables/` | Organiser guidelines and team reports |

Use `train` for model fitting and tuning, and `student_test` for development evaluation. Reserve evaluator-testing trips for final scoring. The [team SOP](SOP.md) covers the remaining integration and submission requirements, and the [model ownership notes](MODEL_OWNERSHIP.md) list team responsibilities.
