# Patch-wise Kernel SHAP for MRI radiomics

This repository contains the analysis code, example data, and trained models used to produce spatial explanations for a radiomics-based support vector machine (SVM). The method assigns importance to three-dimensional regions of an enhancing tumour by measuring changes in model prediction when recipient image patches are replaced with registered image patches from donor cases.

The repository accompanies a manuscript. It is intended to document the analysis and support reproducibility; it is not a clinical software package or medical device.

## Method overview

For a selected recipient case, the analysis:

1. loads standardized post-contrast T1-weighted and T2-FLAIR MRI volumes and an enhancing-tumour mask;
2. partitions the tumour region into fixed-size three-dimensional patches;
3. rigidly registers each donor mask to the recipient mask and resamples the donor images into recipient space;
4. constructs perturbed cases by retaining recipient patches or replacing them with corresponding registered donor image patches;
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
│       ├── models/             Trained cross-validation models
│       └── feature_columns*    Ordered model inputs for each fold
├── radiomics/
│   └── params.yaml             PyRadiomics configuration
├── src/
│   └── run_patchwise_kernel_shap.py
├── requirements.txt
└── README.md
```

## Example data

Three example cases are included. Each case has the following files:

```text
<subject_id>_T1POST_STD.nii
<subject_id>_T2FLAIRPOST_STD.nii
<subject_id>_ET_MASK.nii
```

The images are brain extracted, spatially registered, and intensity standardized within subject. Non-zero mask voxels identify the enhancing tumour.

## Installation and use

Setup instructions, the example command, input requirements, and output descriptions are in [docs/usage.md](docs/usage.md).

## Models and features

Five outer-fold SVM models are provided with their ordered input metadata. Each model uses 186 radiomic features: 93 from T2-FLAIR and 93 from post-contrast T1-weighted MRI.

## Citation

If you use this repository, cite the accompanying manuscript, *Explainable Artificial Intelligence Using Cooperative Game Theory Reveals Tumour Composition in Treated Brain Metastases*.
