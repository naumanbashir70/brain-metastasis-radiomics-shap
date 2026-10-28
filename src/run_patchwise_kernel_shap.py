"""
Patch-wise Kernel SHAP analysis for a radiomics SVM.

The case being explained is the recipient. Each tumour patch is treated as a
feature and is either retained or replaced with the corresponding patch from a
registered donor case. The donor-imputation scheme was developed by Chris Heyn
(2026).

Input images must be registered, brain extracted, and intensity standardized.
Required files follow the naming convention documented in the repository
README. The analysis writes registered donor images, a patch-index image,
patch-wise SHAP maps, and a JSON summary.

Coalition values are encoded as follows: 1 retains the recipient patch, while
values 2 through B+1 select donor rows 0 through B-1.

Example:
python run_patchwise_kernel_shap.py \
  --recipient_id DSC_0153 \
  --bkg_donor_ids DSC_0003 DSC_0012 \
  --data_dir ~/RAD_NEC_RADIOMICS/data/images \
  --out_dir  ~/RAD_NEC_RADIOMICS/outputs \
  --channels t2 t1 \
  --patch 5 5 5 \
  --fold 5 \
  --seed 42 \
  --radiomics_yaml ~/RAD_NEC_RADIOMICS/radiomics/params.yaml \
  --allow_rotation \
  --nsamples 5000
"""

from __future__ import annotations
import os
import json
import time
import glob
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
import joblib
from radiomics import featureextractor
import shap
from sklearn.base import BaseEstimator, TransformerMixin


# Required when loading the serialized training pipeline.

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove correlated features using statistics from the training set."""
    def __init__(self, threshold=0.9, method="spearman"):
        self.threshold = float(threshold)
        self.method = str(method)
        self.kept_indices_ = None

    def fit(self, X, y=None):
        if X is None:
            self.kept_indices_ = []
            return self
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0:
            self.kept_indices_ = []
            return self
        Xdf = pd.DataFrame(X)
        corr = Xdf.corr(method="spearman" if self.method.lower() == "spearman" else "pearson")
        corr = corr.abs().fillna(0.0)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            if (upper[col] > self.threshold).any():
                to_drop.add(col)
        n_cols = Xdf.shape[1]
        self.kept_indices_ = [i for i in range(n_cols) if i not in to_drop]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            return X
        if not hasattr(self, "kept_indices_") or self.kept_indices_ is None:
            return X
        idx = np.asarray(self.kept_indices_, dtype=int)
        if idx.size == 0:
            return X[:, :0]
        idx = idx[(idx >= 0) & (idx < X.shape[1])]
        if idx.size == 0:
            return X[:, :0]
        return X[:, idx]

# I/O and array conversion

def expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def nii_path(data_dir: str, pid: str, suffix: str) -> str:
    return str(Path(data_dir) / f"{pid}_{suffix}.nii")

def load_nib(path: str):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)  # nibabel uses X, Y, Z order
    return img, data

def save_nib_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: str, dtype=np.float32):
    data = np.asarray(data, dtype=dtype)
    out = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    nib.save(out, out_path)

def sitk_read(path: str) -> sitk.Image:
    return sitk.ReadImage(path)

def sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)  # SimpleITK returns Z, Y, X order

def xyz_to_sitk_like(ref: sitk.Image, arr_xyz: np.ndarray, pixel_id=sitk.sitkFloat32) -> sitk.Image:
    """Convert an X, Y, Z array to a SimpleITK image in reference geometry."""
    arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
    out = sitk.GetImageFromArray(arr_zyx)
    out.CopyInformation(ref)
    out = sitk.Cast(out, pixel_id)
    return out

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

def logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p / (1 - p)))

# Patch construction

def tumor_patches(mask_xyz: np.ndarray, patch_xyz: tuple[int,int,int]) -> list[tuple[int,int,int]]:
    px, py, pz = patch_xyz
    xs, ys, zs = np.where(mask_xyz > 0)
    if xs.size == 0:
        return []

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    z0, z1 = zs.min(), zs.max()

    starts = []
    for z in range(int(z0 // pz) * pz, int(z1 // pz) * pz + 1, pz):
        for y in range(int(y0 // py) * py, int(y1 // py) * py + 1, py):
            for x in range(int(x0 // px) * px, int(x1 // px) * px + 1, px):
                if x < 0 or y < 0 or z < 0:
                    continue
                if x + px > mask_xyz.shape[0] or y + py > mask_xyz.shape[1] or z + pz > mask_xyz.shape[2]:
                    continue
                if np.any(mask_xyz[x:x+px, y:y+py, z:z+pz] > 0):
                    starts.append((x,y,z))
    return starts

def make_patch_index_mask(shape_xyz: tuple[int,int,int], patch_xyz: tuple[int,int,int], patch_starts: list[tuple[int,int,int]]) -> np.ndarray:
    px, py, pz = patch_xyz
    out = np.zeros(shape_xyz, dtype=np.int32)
    for j, (x,y,z) in enumerate(patch_starts):
        out[x:x+px, y:y+py, z:z+pz] = j + 1  # Patch IDs are 1 through M.
    return out

# Model and radiomics utilities

def find_fold_artifacts(out_dir: str, channels: list[str], fold: int, seed: int | None = None) -> tuple[str,str]:
    ch = "_".join(channels)
    svm_dir = Path(out_dir) / f"SVM_{ch}"
    model_dir = svm_dir / "models"
    if not svm_dir.exists():
        raise FileNotFoundError(f"SVM directory not found: {svm_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    seed_pat = f"*seed={seed}*" if seed is not None else "*"

    m_globs = [
        str(model_dir / f"svm_model*{seed_pat}*outer={fold}*.joblib"),
        str(model_dir / f"*{seed_pat}*outer={fold}*.joblib"),
    ]
    m_hits = []
    for g in m_globs:
        m_hits.extend(sorted(glob.glob(g)))
    m_hits = sorted(set(m_hits))
    if not m_hits:
        raise FileNotFoundError("No model matched any of:\n  " + "\n  ".join(m_globs))

    f_globs = [
        str(svm_dir / f"feature_columns*{seed_pat}*outer={fold}*.csv"),
        str(svm_dir / f"feature_columns*{seed_pat}*outer={fold}*.json"),
        str(svm_dir / f"*feature_columns*{seed_pat}*outer={fold}*.csv"),
        str(svm_dir / f"*feature_columns*{seed_pat}*outer={fold}*.json"),
        str(svm_dir / f"test_predictions*{seed_pat}*outer={fold}*.csv"),
        str(svm_dir / f"*test_predictions*{seed_pat}*outer={fold}*.csv"),
    ]
    f_hits = []
    for g in f_globs:
        f_hits.extend(sorted(glob.glob(g)))
    f_hits = sorted(set(f_hits))
    if not f_hits:
        raise FileNotFoundError("No feature-columns file matched any of:\n  " + "\n  ".join(f_globs))

    return m_hits[0], f_hits[0]

def load_feature_columns(path: str) -> list[str]:
    p = path.lower()
    if p.endswith(".json"):
        with open(path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict) and "features" in obj and isinstance(obj["features"], list):
            return [str(x) for x in obj["features"]]
        return [str(k) for k in obj.keys()]

    if p.endswith(".csv"):
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            return df.iloc[:, 0].astype(str).tolist()
        for c in ["feature", "features", "name"]:
            if c in df.columns:
                return df[c].astype(str).tolist()

        drop_candidates = {
            "patient_id","subject_id","id",
            "y","y_true","label","target",
            "pred","pred_label","prediction",
            "proba","prob","probability",
            "fold","outer","inner","seed",
        }
        cols = [c for c in df.columns if c not in drop_candidates]
        if len(cols) == 0:
            cols = list(df.columns)
        return [str(c) for c in cols]

    raise ValueError(f"Unsupported feature columns file: {path}")

def radiomics_from_sitk(
    extractor: featureextractor.RadiomicsFeatureExtractor,
    img_sitk: sitk.Image,
    mask_sitk: sitk.Image,
    prefix: str,
) -> dict:
    res = extractor.execute(img_sitk, mask_sitk)
    feats = {}
    for k, v in res.items():
        if k.startswith("diagnostics"):
            continue
        try:
            feats[prefix + k] = float(v)
        except Exception:
            pass
    return feats

def predict_probability(model, feature_cols: list[str], feat_dict: dict) -> float:
    x = np.zeros((1, len(feature_cols)), dtype=np.float32)
    for i, name in enumerate(feature_cols):
        if name in feat_dict:
            x[0, i] = feat_dict[name]
        # Features absent from the extracted set retain their zero value.
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x)[0, 1])
    score = float(model.decision_function(x)[0])
    return sigmoid(score)

# Donor-to-recipient registration

def rigid_register_mask(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    allow_rotation: bool = True,
    max_iters: int = 200,
) -> sitk.Transform:
    fixed = sitk.Cast(fixed_mask, sitk.sitkFloat32)
    moving = sitk.Cast(moving_mask, sitk.sitkFloat32)

    if not allow_rotation:
        initial = sitk.TranslationTransform(3)
        fixed_stats = sitk.LabelShapeStatisticsImageFilter()
        moving_stats = sitk.LabelShapeStatisticsImageFilter()
        fixed_stats.Execute(sitk.Cast(fixed_mask > 0, sitk.sitkUInt8))
        moving_stats.Execute(sitk.Cast(moving_mask > 0, sitk.sitkUInt8))
        if fixed_stats.GetNumberOfLabels() > 0 and moving_stats.GetNumberOfLabels() > 0:
            fl = fixed_stats.GetLabels()[0]
            ml = moving_stats.GetLabels()[0]
            cf = np.array(fixed_stats.GetCentroid(fl))
            cm = np.array(moving_stats.GetCentroid(ml))
            initial.SetOffset((cf - cm).tolist())

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetMetricSamplingStrategy(reg.NONE)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0, minStep=1e-4, numberOfIterations=max_iters,
            gradientMagnitudeTolerance=1e-6, relaxationFactor=0.5,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetInitialTransform(initial, inPlace=False)
        return reg.Execute(fixed, moving)

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=max_iters,
        gradientMagnitudeTolerance=1e-6, relaxationFactor=0.5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)
    return reg.Execute(fixed, moving)

def resample_to_fixed(fixed_ref: sitk.Image, moving_img: sitk.Image, transform: sitk.Transform, is_mask: bool) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    out = sitk.Resample(moving_img, fixed_ref, transform, interp, 0, moving_img.GetPixelID())
    if is_mask:
        out = sitk.Cast(out > 0.5, sitk.sitkUInt8)
    return out

def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else 0.0

# Patch imputation

def apply_assign_vec(
    rec_t1: np.ndarray,
    rec_t2: np.ndarray,
    rec_m: np.ndarray,
    donors: list[dict],
    patch_xyz: tuple[int,int,int],
    patch_starts: list[tuple[int,int,int]],
    assign_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply the recipient/donor assignment for each patch."""
    px, py, pz = patch_xyz
    M = len(patch_starts)
    if assign_vec.shape[0] != M:
        raise ValueError(f"assign_vec length {assign_vec.shape[0]} != M={M}")

    out_t1 = rec_t1.copy()
    out_t2 = rec_t2.copy()
    out_m  = rec_m.copy()

    usage = {
        "keep": 0,
        "by_donor": {d["donor_id"]: 0 for d in donors},
        "patches_by_donor": {d["donor_id"]: [] for d in donors},
    }

    for j, (x,y,z) in enumerate(patch_starts):
        v = int(assign_vec[j])
        if v == 1:
            usage["keep"] += 1
            continue
        di = v - 2
        if di < 0 or di >= len(donors):
            raise ValueError(f"assign_vec[{j}]={v} refers to donor index {di}, but B={len(donors)}")
        d = donors[di]
        out_t1[x:x+px, y:y+py, z:z+pz] = d["t1_xyz"][x:x+px, y:y+py, z:z+pz]
        out_t2[x:x+px, y:y+py, z:z+pz] = d["t2_xyz"][x:x+px, y:y+py, z:z+pz]
        out_m [x:x+px, y:y+py, z:z+pz] = d["m_xyz"][x:x+px, y:y+py, z:z+pz]
        usage["by_donor"][d["donor_id"]] += 1
        usage["patches_by_donor"][d["donor_id"]].append(j)

    return out_t1, out_t2, out_m, usage


# Analysis entry point

def main():
    ap = argparse.ArgumentParser(description="KernelSHAP with patch-wise donor-imputation features.")
    ap.add_argument("--recipient_id", required=True)
    ap.add_argument("--bkg_donor_ids", nargs="+", required=True, help="B donors used as background rows (B=len(list))")
    ap.add_argument("--data_dir", default="~/RAD_NEC_RADIOMICS/data/images")
    ap.add_argument("--out_dir",  default="~/RAD_NEC_RADIOMICS/outputs")
    ap.add_argument("--channels", nargs="+", default=["t2","t1"])
    ap.add_argument("--patch", nargs=3, type=int, default=[5,5,5])
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radiomics_yaml", required=True)
    ap.add_argument("--allow_rotation", action="store_true")
    ap.add_argument("--reg_iters", type=int, default=200)
    ap.add_argument("--nsamples", type=int, default=200)
    ap.add_argument("--l1_reg", type=str, default="num_features(50)")
    args = ap.parse_args()

    t0_all = time.time()
    rng = np.random.default_rng(args.seed)

    data_dir = expand(args.data_dir)
    out_dir  = expand(args.out_dir)
    rad_yaml = expand(args.radiomics_yaml)

    rid = args.recipient_id
    donor_ids = args.bkg_donor_ids
    channels = [c.lower() for c in args.channels]
    patch_xyz = tuple(int(x) for x in args.patch)

    rec_t1_p = nii_path(data_dir, rid, "T1POST_STD")
    rec_t2_p = nii_path(data_dir, rid, "T2FLAIRPOST_STD")
    rec_m_p  = nii_path(data_dir, rid, "ET_MASK")
    for p in [rec_t1_p, rec_t2_p, rec_m_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    ch_tag = "_".join(channels)
    donors_tag = "+".join(donor_ids)
    run_dir = Path(out_dir) / f"{rid}_KSHAP_IMPUTE_B{len(donor_ids)}_{donors_tag}_CH{ch_tag}_P{patch_xyz[0]}x{patch_xyz[1]}x{patch_xyz[2]}_F{args.fold}_S{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep nibabel arrays for patch operations and SimpleITK geometry for registration.
    rec_t1_nib, rec_t1_xyz = load_nib(rec_t1_p)
    rec_t2_nib, rec_t2_xyz = load_nib(rec_t2_p)
    _, rec_m_xyz_f = load_nib(rec_m_p)
    rec_m_xyz = (rec_m_xyz_f > 0.5).astype(np.uint8)

    rec_m_sitk = sitk_read(rec_m_p)
    rec_m_sitk = sitk.Cast(rec_m_sitk > 0.5, sitk.sitkUInt8)

    patch_starts = tumor_patches(rec_m_xyz, patch_xyz)
    if len(patch_starts) == 0:
        raise RuntimeError(f"No tumor-intersecting patches found for recipient {rid} with patch={patch_xyz}")
    Mdim = len(patch_starts)

    patch_index = make_patch_index_mask(rec_t1_xyz.shape, patch_xyz, patch_starts)
    patch_index_p = str(run_dir / f"{rid}_PATCH_INDEX_P{patch_xyz[0]}x{patch_xyz[1]}x{patch_xyz[2]}.nii")
    save_nib_like(rec_t1_nib, patch_index.astype(np.int32), patch_index_p, dtype=np.int32)

    model_path, featcols_path = find_fold_artifacts(out_dir, channels, args.fold, seed=args.seed)
    model = joblib.load(model_path)
    feature_cols = load_feature_columns(featcols_path)

    extractor = featureextractor.RadiomicsFeatureExtractor(rad_yaml) 

    # Register donor masks, then apply the same transform to both MRI channels.
    donors = []
    dice_by_donor = {}
    for did in donor_ids:
        don_t1_p = nii_path(data_dir, did, "T1POST_STD")
        don_t2_p = nii_path(data_dir, did, "T2FLAIRPOST_STD")
        don_m_p  = nii_path(data_dir, did, "ET_MASK")
        for p in [don_t1_p, don_t2_p, don_m_p]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing: {p}")

        don_m_sitk = sitk_read(don_m_p)
        don_m_sitk = sitk.Cast(don_m_sitk > 0.5, sitk.sitkUInt8)

        tx = rigid_register_mask(rec_m_sitk, don_m_sitk, allow_rotation=args.allow_rotation, max_iters=args.reg_iters)

        don_t1_sitk = sitk_read(don_t1_p)
        don_t2_sitk = sitk_read(don_t2_p)

        don_m_reg  = resample_to_fixed(rec_m_sitk, don_m_sitk,  tx, is_mask=True)
        don_t1_reg = resample_to_fixed(rec_m_sitk, don_t1_sitk, tx, is_mask=False)
        don_t2_reg = resample_to_fixed(rec_m_sitk, don_t2_sitk, tx, is_mask=False)

        don_m_xyz  = np.transpose(sitk_to_numpy(don_m_reg),  (2,1,0)).astype(np.uint8)
        don_t1_xyz = np.transpose(sitk_to_numpy(don_t1_reg), (2,1,0)).astype(np.float32)
        don_t2_xyz = np.transpose(sitk_to_numpy(don_t2_reg), (2,1,0)).astype(np.float32)

        dice_by_donor[did] = dice(rec_m_xyz, don_m_xyz)

        # Save transformed inputs for registration quality control.
        save_nib_like(rec_t1_nib, don_t1_xyz, str(run_dir / f"{did}_T1POST_STD_XFM_TO_{rid}.nii"))
        save_nib_like(rec_t2_nib, don_t2_xyz, str(run_dir / f"{did}_T2FLAIRPOST_STD_XFM_TO_{rid}.nii"))
        save_nib_like(rec_t1_nib, don_m_xyz.astype(np.uint8), str(run_dir / f"{did}_ET_MASK_XFM_TO_{rid}.nii"), dtype=np.uint8)

        donors.append({
            "donor_id": did,
            "t1_xyz": don_t1_xyz,
            "t2_xyz": don_t2_xyz,
            "m_xyz":  don_m_xyz,
        })

    # Prediction for the unmodified recipient.
    rec_t1_s = xyz_to_sitk_like(rec_m_sitk, rec_t1_xyz, pixel_id=sitk.sitkFloat32)
    rec_t2_s = xyz_to_sitk_like(rec_m_sitk, rec_t2_xyz, pixel_id=sitk.sitkFloat32)
    rec_m_s  = xyz_to_sitk_like(rec_m_sitk, rec_m_xyz.astype(np.uint8), pixel_id=sitk.sitkUInt8)

    feats_full = {}
    for ch in channels:
        if ch == "t1":
            feats_full.update(radiomics_from_sitk(extractor, rec_t1_s, rec_m_s, "t1__"))
        elif ch == "t2":
            feats_full.update(radiomics_from_sitk(extractor, rec_t2_s, rec_m_s, "t2__"))
        else:
            raise ValueError("channels must be subset/order of: t1 t2")

    p_full = predict_probability(model, feature_cols, feats_full)
    full_logit = logit(p_full)

    # Each background row represents a complete imputation from one donor.
    B = len(donors)
    background_keep = np.zeros((B, Mdim), dtype=int) 

    background_keep_encoded = background_keep.astype(float)
    for r in range(B):
        tag = float(r + 2)
        background_keep_encoded[r, background_keep_encoded[r] == 0] = tag

    x_instance = np.ones(Mdim, dtype=int)

    cache = {}  # Coalition assignment bytes mapped to predicted probability.

    def f_coalitions(Z):
        """Return model probabilities for a batch of patch assignments."""
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z[None, :]

        out = np.zeros((Z.shape[0],), dtype=np.float64)

        for i in range(Z.shape[0]):
            v = Z[i, :].astype(int)

            key = v.tobytes()
            if key in cache:
                out[i] = cache[key]
                continue

            imp_t1, imp_t2, imp_m, _usage = apply_assign_vec(
                rec_t1_xyz.astype(np.float32),
                rec_t2_xyz.astype(np.float32),
                rec_m_xyz.astype(np.uint8),
                donors,
                patch_xyz,
                patch_starts,
                v
            )

            imp_m_s = xyz_to_sitk_like(rec_m_sitk, imp_m.astype(np.uint8), pixel_id=sitk.sitkUInt8)
            feats = {}

            for ch in channels:
                if ch == "t1":
                    imp_t1_s = xyz_to_sitk_like(rec_m_sitk, imp_t1, pixel_id=sitk.sitkFloat32)
                    feats.update(radiomics_from_sitk(extractor, imp_t1_s, imp_m_s, "t1__"))
                elif ch == "t2":
                    imp_t2_s = xyz_to_sitk_like(rec_m_sitk, imp_t2, pixel_id=sitk.sitkFloat32)
                    feats.update(radiomics_from_sitk(extractor, imp_t2_s, imp_m_s, "t2__"))

            p = predict_probability(model, feature_cols, feats)
            cache[key] = p
            out[i] = p

        return out
    print(f"[INFO] recipient={rid} | B donors={B} | M patches={Mdim}")
    print(f"[INFO] full probability p_full={p_full:.6f}")
    print(f"[INFO] running KernelExplainer(nsamples={args.nsamples}, link='logit') ...")

    t0 = time.time()
    explainer = shap.KernelExplainer(f_coalitions, background_keep_encoded, link="logit")
    shap_values = explainer.shap_values(
        x_instance,
        nsamples=max(int(args.nsamples), 1),
        l1_reg=args.l1_reg
    )
    t1 = time.time()

    # Older SHAP releases may return one array per class.
    if isinstance(shap_values, list):
        phi = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0], dtype=float)
    else:
        phi = np.asarray(shap_values, dtype=float)

    expected_logit = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(np.asarray(explainer.expected_value).ravel()[0])
    expected_prob = sigmoid(expected_logit)

    sum_phi = float(np.sum(phi))
    recon_logit = expected_logit + sum_phi
    recon_prob = sigmoid(recon_logit)

    px, py, pz = patch_xyz
    shap_logit_map = np.zeros(rec_t1_xyz.shape, dtype=np.float32)
    shap_prob_effect_map = np.zeros(rec_t1_xyz.shape, dtype=np.float32)

    patch_details = []
    for j, (x,y,z) in enumerate(patch_starts):
        phi_j = float(phi[j])
        # This probability-scale value is descriptive and is not additive.
        prob_effect = sigmoid(expected_logit + phi_j) - expected_prob

        shap_logit_map[x:x+px, y:y+py, z:z+pz] = phi_j
        shap_prob_effect_map[x:x+px, y:y+py, z:z+pz] = float(prob_effect)

        patch_details.append({
            "patch_id": int(j + 1),
            "start_xyz": [int(x), int(y), int(z)],
            "end_xyz_exclusive": [int(x + px), int(y + py), int(z + pz)],
            "centroid_xyz": [float(x + px/2.0), float(y + py/2.0), float(z + pz/2.0)],
            "shap_logit": phi_j,
            "shap_prob_effect_approx": float(prob_effect),
            "donor_pool_codes": {
                "2": donor_ids[0] if B >= 1 else None,
                "3": donor_ids[1] if B >= 2 else None,
                "4": donor_ids[2] if B >= 3 else None,
                "...": "code = r+2 -> bkg_donor_ids[r]"
            }
        })

    shap_logit_p = str(run_dir / f"{rid}_SHAP_LOGIT_P{patch_xyz[0]}x{patch_xyz[1]}x{patch_xyz[2]}.nii")
    shap_prob_eff_p = str(run_dir / f"{rid}_SHAP_PROB_EFFECT_P{patch_xyz[0]}x{patch_xyz[1]}x{patch_xyz[2]}.nii")
    save_nib_like(rec_t1_nib, shap_logit_map.astype(np.float32), shap_logit_p)
    save_nib_like(rec_t1_nib, shap_prob_effect_map.astype(np.float32), shap_prob_eff_p)

    summary = {
        "recipient_id": rid,
        "bkg_donor_ids": donor_ids,
        "B": int(B),
        "M_patches": int(Mdim),
        "channels": channels,
        "patch_xyz": list(patch_xyz),
        "seed": int(args.seed),
        "nsamples": int(args.nsamples),
        "l1_reg": str(args.l1_reg),
        "allow_rotation": bool(args.allow_rotation),
        "reg_iters": int(args.reg_iters),
        "dice_post_registration_by_donor": dice_by_donor,
        "model_path": model_path,
        "feature_columns_source_path": featcols_path,

        "prob_full": float(p_full),
        "logit_full": float(full_logit),

        "expected_logit": float(expected_logit),
        "expected_prob": float(expected_prob),

        "sum_shap_logit": float(sum_phi),
        "expected_plus_sum_logit": float(recon_logit),
        "expected_plus_sum_prob": float(recon_prob),

        "additivity_check_logit": {
            "full_logit": float(full_logit),
            "reconstructed_logit": float(recon_logit),
            "abs_diff": float(abs(full_logit - recon_logit)),
        },

        "runtime_sec": float(t1 - t0),
        "total_runtime_sec": float(time.time() - t0_all),

        "outputs": {
            "run_dir": str(run_dir),
            "patch_index_nii": patch_index_p,
            "shap_logit_nii": shap_logit_p,
            "shap_prob_effect_nii": shap_prob_eff_p,
        },

        "patches": patch_details,
    }

    json_path = run_dir / "shap_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DONE ===")
    print(f"Run dir: {run_dir}")
    print(f"Saved: {json_path}")
    print(f"p_full={p_full:.6f} | expected_prob={expected_prob:.6f} | recon_prob={recon_prob:.6f}")
    print(f"logit additivity abs diff = {abs(full_logit - recon_logit):.6e}")
    print(f"Patch index NIfTI: {patch_index_p}")
    print(f"SHAP logit NIfTI:  {shap_logit_p}")
    print(f"SHAP prob-effect:  {shap_prob_eff_p}")


if __name__ == "__main__":
    main()
