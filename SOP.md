# PathSnap Team 12 Standard Operating Procedure

## Workspace

Work from `E:\ML_Hackathon`. Keep organiser documents in `Requirement Docs\` and the supplied data in `Student folder\`. Keep the original inputs intact and save derived files separately.

GitHub repository [PathSnap_Team12](https://github.com/Krish-Nahhh/PathSnap_Team12).

## 1. Connect the local folder to GitHub

Run these PowerShell commands once from the project folder while the GitHub repository is empty.

```powershell
Set-Location -LiteralPath 'E:\ML_Hackathon'
git init -b main
git remote add origin https://github.com/Krish-Nahhh/PathSnap_Team12.git
```

Before staging anything, create a root `.gitignore` with the rules below. This keeps the supplied files, local environments and credentials out of the repository.

```gitignore
/Student folder/
/Requirement Docs/
/.venv/
__pycache__/
*.py[cod]
.env
.env.*
!.env.example
```

Start with the SOP and ignore rules. Review the staged names before committing.

```powershell
git add -- SOP.md .gitignore
git diff --cached --name-only
git commit -m "Initialize PathSnap Team 12 repository"
git push -u origin main
```

## 2. Inspect the development data

- Inventory the 80 training trips and 10 student-testing trips. Check file availability, timestamps, coordinate ranges, road connectivity and reference arc IDs.
- Verify the semantics of `.track`, `.nodes`, `.arcs` and `.route` against the actual files. Document how reference routes are aligned to GPS observations.
- Record data-quality findings and retain code that can regenerate tables and Folium visualisations. Reserve the 10 evaluator trips for final evaluation.

## 3. Train, compare and tune

- Fit preprocessing and models with the training partition. Group cross-validation folds by trip.
- Train and compare at least five models. Use the student-testing partition for development evaluation. Record configurations, seeds, metrics, timings and the reason for selecting the final model.
- Run Bayesian optimisation on the training partition and retain the search history. Compare the tuned and untuned model on student-testing using the same metrics.
- Implement HMM/Viterbi decoding and compare it with classifier-only predictions. Record pointwise accuracy, sequence performance and route continuity.
- Export the fitted model, preprocessing components and decoder configuration needed for inference.

## 4. Integrate and rehearse

- Run the FastAPI service and evaluator UI on separate laptops. The UI sends a selected trip ID and displays the returned route and scores.
- Verify the full request flow and capture API logs and UI screenshots. Use development trips for rehearsal.
- During final evaluation, score predictions server-side after inference. Keep the evaluation reference labels out of the prediction pipeline and UI.

## 5. Submit

- Check that the repository includes source code, dependencies, configurations, model artifacts, experiment evidence and reproducible run instructions. Review tracked files before making the repository public for submission.
- Prepare one technical report PDF using measured results. Include model comparisons, tuning history, decoder comparisons, a representative Folium map and evidence of the working API and UI.
- Check the latest organiser documents in `Requirement Docs\` before submitting.
