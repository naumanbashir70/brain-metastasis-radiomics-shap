# Patch-wise Kernel SHAP for MRI radiomics

This repository contains the analysis code and example data used to produce spatial explanations for a radiomics-based support vector machine (SVM). The method assigns importance to three-dimensional patches of an enhancing tumour by measuring how the model prediction changes when recipient patches are replaced with registered patches from donor cases.

The repository accompanies a manuscript. It is intended to document the analysis and support reproducibility; it is not a clinical software package or medical device.

## Method overview

For a selected recipient case, the analysis:

1. loads standardized post-contrast T1-weighted and T2-FLAIR MRI volumes and an enhancing-tumour mask;
2. partitions the tumour region into fixed-size three-dimensional patches;
3. rigidly registers each donor mask to the recipient mask and resamples the donor images into recipient space;
4. constructs perturbed cases by retaining recipient patches or replacing them with corresponding donor patches;
5. extracts radiomic features from each perturbed case and obtains an SVM probability;
6. applies Kernel SHAP to estimate the contribution of each recipient patch; and
7. writes patch-level attribution maps and a JSON record of the analysis.

Further methodological detail is provided in [docs/method.md](docs/method.md).

## Repository structure

```text
.
├── data/
│   └── images/                 Example MRI volumes and tumour masks
├── docs/                       Method and usage documentation
├── outputs/
│   └── SVM_t2_t1/
│       └── models/             Trained cross-validation models
├── radiomics/
│   └── params.yaml             PyRadiomics configuration
├── src/
│   └── run_patchwise_kernel_shap.py
├── requirements.txt
└── README.md
```

The analysis is provided as a single script to preserve the workflow used for the manuscript.

## Example data

Three example cases are included. Each case has the following files:

```text
<subject_id>_T1POST_STD.nii
<subject_id>_T2FLAIRPOST_STD.nii
<subject_id>_ET_MASK.nii
```

The images are expected to be brain extracted, spatially registered, and intensity standardized within subject. The mask must identify the enhancing tumour using non-zero voxels.

Before making the repository public, the data owner should independently confirm that every image is appropriately de-identified and approved for public distribution.

## Installation and use

Setup instructions, the example command, input requirements, and output descriptions are in [docs/usage.md](docs/usage.md).

## Reproducibility note

The current repository includes five trained SVM models but does not include the corresponding feature-column metadata required by the analysis script. For each fold, the script expects a matching `feature_columns` JSON/CSV file or a compatible `test_predictions` CSV file in `outputs/SVM_t2_t1/`. These files must be restored from the model-training analysis before the example can run end to end. No substitute feature order should be inferred, because the trained pipeline depends on the original column order.

## Data and model storage

NIfTI images and serialized models are configured for [Git Large File Storage](https://git-lfs.com/). Install Git LFS before adding the repository to Git. The original ZIP export is intentionally excluded from version control because it duplicates the repository and exceeds GitHub's normal per-file limit.

## Citation

If you use this repository, cite the accompanying manuscript. Full citation details and a `CITATION.cff` file should be added when the manuscript bibliographic record is finalized.

## License

No software or data license has been assigned. Until the authors add explicit terms, the contents remain subject to the rights of their respective owners and should not be treated as open-source or open data.
