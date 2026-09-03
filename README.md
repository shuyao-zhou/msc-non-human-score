# Quantifying Automated Behaviour in Enterprise Authentication Logs

Code accompanying the MSc dissertation on an unsupervised continuous
Non-Human Score (NHS) for enterprise authentication activity.

## Notebook order

Run the analysis in the following order:

1. `00_raw_behaviour_eda.ipynb`
2. `01_build_static_features.ipynb`
3. `01b_feature_assessment_CORE5.ipynb`
4. `02a_custom_mixture_development.ipynb`
5. `02b_core5_model_comparison.ipynb`
6. `02c_final_static_test_evaluation.ipynb`
7. `03_model_disagreement.ipynb`
8. `03b_manual_behavioural_inspection.ipynb`
9. `04_machine_proxy_behaviour_injection.ipynb`

The supporting modules are:

- `nhs_feature_utils.py`
- `nhs_split_eval.py`
- `nhs_mixture.py`

## Data

The Los Alamos National Laboratory authentication data are not redistributed
in this repository.

The notebooks currently use the authentication source path configured in their
configuration cells. Change `AUTH_PATH` to the local location of the LANL
authentication file before rebuilding the raw-event cache.

Large generated data files such as `auth_main.parquet` and the static feature
Parquet tables are intentionally excluded from version control.

## Fixed split and reusable intermediate results

The pipeline uses a fixed account-level fit/calibration/test split.

Small reproducibility artefacts may be committed with the repository,
including the split metadata, quiet-hour definition, CORE5 representation
metadata, frozen custom-mixture state and frozen generic scorer state.

Precomputed intermediate outputs from computationally expensive mixture
fitting are included for convenience. The notebooks retain the code required
to regenerate these outputs from the fitting partition.

`02a_custom_mixture_development.ipynb` reuses the saved BIC scan and frozen
K=6 state when they are available. If they are absent, the notebook contains
the code required to refit them.

## Custom-mixture implementation

The K-component custom mixture is identified by
`MIXTURE_MODEL_VERSION` in `nhs_mixture.py`. The frozen mixture state records
the same identifier, and downstream notebooks check it before scoring.

## Evaluation references

Account-name rules are used only as weak reference groups and not as verified
ground-truth account types. The final scorer comparison evaluates the broader
automation-associated reference group (dollar-suffix accounts together with
SYSTEM, LOCAL SERVICE and NETWORK SERVICE) against U-named user proxies.

## Reproducibility

The final static pipeline freezes scorer-specific fitting quantities before
calibration/test application. The selected K-means scorer is not refitted in
the final test evaluation, disagreement analysis, manual behavioural review,
or semi-synthetic responsiveness experiment.
