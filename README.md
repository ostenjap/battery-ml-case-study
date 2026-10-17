# PaaS4Bat PINN Case Study — Battery Capacity Fade Prediction


Physics-Informed Neural Network (PINN) that predicts battery capacity fade over charge/discharge cycles, benchmarked against ARIMA and LSTM baselines.

## Repo layout

- `documents/` -- the source-of-truth notebooks, run in order:
  data extraction/cleaning, the MLP and LSTM baselines, the step-3
  comparison table, the SEI-growth PINN, and the two-channel recovery PINN.
- `data/` -- `raw/` (downloaded NASA `.mat` files) and `processed/`
  (`clean_data.csv`, the one file every notebook trains on).
- `src/` -- the plain Python modules the notebooks import (data fetching,
  extraction, the synthetic warm-up PINN sweep, the report's figure script)
  so logic isn't duplicated across notebooks.
- `reports/` -- the technical report (`technical_report.pdf`).
- `models/` -- saved model checkpoints (`.pt`) from exploratory runs.
- `logs/` -- local run logs and a session-tracking database; not tracked
  in git.
- `docs/` -- design decisions (`decisions.md`) and earlier/exploratory
  notebooks that were superseded by the ones in `documents/`.
- `notebooks/` -- an early prototyping notebook, superseded by `documents/`.
- `NotRelevant/` -- earlier planning docs kept for reference, not part of
  the current project.

