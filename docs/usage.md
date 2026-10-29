# Usage

## System requirements

Use Python 3.10 or later. Image registration and repeated radiomic feature extraction can be computationally expensive; runtime increases with the number of tumour patches and SHAP samples.

Git LFS is required when cloning or publishing the included NIfTI images and model files.

## Installation

Clone the repository, create an isolated environment, and install the Python dependencies:

```bash
git lfs install
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file records direct packages used by the script. For exact reproduction of manuscript results, use the package versions from the original analysis environment if they are available.

## Required model artifacts

The script searches for two fold-specific artifacts:

```text
outputs/SVM_t2_t1/models/svm_model[t2+t1_seed=42_outer=<fold>].joblib
outputs/SVM_t2_t1/feature_columns[t2+t1_seed=42_outer=<fold>].json
```

Feature columns may also be supplied as CSV, and the script recognizes compatible fold-specific `test_predictions` CSV files. The feature metadata must describe the exact ordered inputs used to train the serialized model.

The model and feature-column files for all five outer folds are included in this repository.

## Example command

Run the command from the repository root:

```bash
python src/run_patchwise_kernel_shap.py \
  --recipient_id DSC_0153 \
  --bkg_donor_ids DSC_0003 DSC_0012 \
  --data_dir data/images \
  --out_dir outputs \
  --channels t2 t1 \
  --patch 5 5 5 \
  --fold 5 \
  --seed 42 \
  --radiomics_yaml radiomics/params.yaml \
  --allow_rotation \
  --nsamples 5000
```

For a short validation run, reduce `--nsamples`; this changes the approximation and is not equivalent to the manuscript analysis.

## Command-line arguments

| Argument | Meaning |
| --- | --- |
| `--recipient_id` | Subject to explain. |
| `--bkg_donor_ids` | One or more donor subjects used as SHAP background cases. |
| `--data_dir` | Directory containing NIfTI images and masks. |
| `--out_dir` | Parent directory for models and generated run directories. |
| `--channels` | MRI channels and feature-prefix order; supported values are `t1` and `t2`. |
| `--patch` | Patch dimensions in voxels, given as X Y Z. |
| `--fold` | Outer cross-validation fold used to select model artifacts. |
| `--seed` | Seed identifier used to select model artifacts and initialize the run. |
| `--radiomics_yaml` | PyRadiomics parameter file. |
| `--allow_rotation` | Use rigid rotation and translation rather than translation only. |
| `--reg_iters` | Maximum registration optimizer iterations. |
| `--nsamples` | Number of Kernel SHAP samples. |
| `--l1_reg` | Kernel SHAP feature-selection regularization setting. |

Use the script's built-in help to view the accepted options:

```bash
python src/run_patchwise_kernel_shap.py --help
```

## Output directory

The run-directory name records the recipient, donor pool, channels, patch dimensions, fold, and seed. Generated directories follow this pattern:

```text
outputs/<recipient>_KSHAP_IMPUTE_B<count>_<donors>_CH<channels>_P<size>_F<fold>_S<seed>/
```

The `shap_summary.json` file is the primary machine-readable record. It includes the full prediction, expected prediction, SHAP sum, additivity error, registration Dice scores, runtime, output paths, and coordinates and attribution values for every patch.

## Adding another case

Place all three subject files in `data/images/` using the established naming convention. Confirm that the two MRI volumes and mask have matching dimensions and physical geometry. The new case may then be supplied as a recipient or donor identifier.

Do not use clinical identifiers in file names or NIfTI headers. Public release should follow the relevant ethics approval, consent conditions, and institutional data-sharing policy.
