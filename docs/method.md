# Method

## Objective

The analysis explains an SVM prediction for a selected tumour by assigning spatial importance to its three-dimensional image patches. Each patch is treated as a SHAP feature. Instead of replacing an absent feature with a constant value, the method uses the corresponding registered MRI and mask patch from a donor case.

## Inputs

The analysis requires post-contrast T1-weighted and T2-FLAIR volumes, an enhancing-tumour mask, a trained SVM pipeline, the ordered feature names used during training, and a PyRadiomics parameter file.

Images are read in NIfTI format. The code assumes that the modalities and mask for a given subject already share geometry and that image intensities have been standardized during preprocessing.

## Patch construction

The bounding region of the recipient tumour is traversed using the requested patch dimensions. A patch is retained when it contains at least one non-zero tumour-mask voxel. Patches are assigned consecutive identifiers, which are written to a patch-index NIfTI image.

## Donor registration and imputation

Each donor tumour mask is rigidly registered to the recipient mask. Registration may use translation alone or an Euler three-dimensional transform when rotation is enabled. The resulting transform is applied to both donor MRI volumes and the donor mask. Linear interpolation is used for MRI data and nearest-neighbour interpolation for the mask.

For each coalition evaluated by Kernel SHAP, every patch is assigned one of two states:

- retain the recipient image and mask values; or
- replace them with the corresponding registered donor image and mask values.

The spatial arrangement of patches is therefore preserved. Registration quality is summarized by the Dice coefficient between the recipient mask and each transformed donor mask.

## Radiomics prediction

PyRadiomics features are extracted independently from each requested MRI channel using the imputed mask. Feature names are prefixed with `t1__` or `t2__` and placed in the original training-column order. The trained model returns a class-one probability through `predict_proba`, or a sigmoid-transformed decision score when probability prediction is unavailable.

The supplied radiomics configuration enables original-image first-order and texture families: GLCM, GLRLM, GLSZM, GLDM, and NGTDM. It uses a bin width of 0.5 and does not normalize intensities during feature extraction.

## Kernel SHAP analysis

Background rows represent complete donor imputations. The instance being explained represents the unmodified recipient. Kernel SHAP evaluates mixed coalitions and estimates a contribution for each tumour patch using a logit link.

The principal attribution is the SHAP value in logit space. The probability-effect image is a per-patch approximation calculated relative to the expected prediction; these values are not additive. The JSON summary records the logit additivity check.

## Outputs

Each run creates a uniquely named directory under `outputs/` containing:

- the recipient patch-index image;
- registered donor T1, T2-FLAIR, and mask images;
- a patch-wise SHAP image in logit space;
- an approximate patch-wise probability-effect image; and
- `shap_summary.json`, containing parameters, paths, registration scores, prediction summaries, attribution values, and runtimes.

These maps preserve the recipient NIfTI affine and header. They should be interpreted with the corresponding patch-index image and the analysis settings recorded in the JSON file.
