# Candidate generation

From the repository root, pass the extracted trip directory to the standard-library CLI.

```powershell
python build_candidates.py generate "E:\ML_Hackathon\Student folder\Student folder\train\00000001" --output results/candidates_00000001.csv
python build_candidates.py score results/candidates_00000001.csv aligned_labels.csv
```

The generator ranks directed arcs by distance to the straight segment between their endpoint nodes, using a local flat-earth projection in metres. The initial settings are `--radius-m 75 --top-k 10`. Ties break by arc ID. GPS heading uses the previous fix and is blank for the first fix, gaps outside (0, 10] seconds, movement below 3 m or speed below 1 m/s. Heading error is the shortest angular difference in degrees.

The [sample export](results/candidate_sample.csv) covers the first five fixes of training trip `00000001`. Candidate CSVs use zero-based `observation_index` and `arc_id` from the source file rows and one-based `candidate_rank`.

For scoring, provide aligned labels with columns `split,trip_id,observation_index,true_arc_id`. Use labels covering the same trip and observation range as the candidate export. The scorer counts a miss when the true directed arc is absent from the candidate list, including observations with zero candidates. It reports the rate across unique labeled observations.
