"""One-fold retrain to confirm the soil_texture_idx log-transform fix unsticks
u_k[Clay Loam=7] and u_k[Clay=11].

Writes weights/manifest to a verification-only directory so the production v8
checkpoints are untouched.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold

PROJECT_ROOT = Path("/Users/giogonzales/Documents/ml-prep/mlprep")
sys.path.insert(0, str(PROJECT_ROOT))

# Reproduce the notebook seed for comparability with the v8 fold-1 split.
from py_files.helpers import set_seed
set_seed(42)

# Monkey-patch StratifiedKFold inside train_rainfall_v3 so only fold 1 runs.
class OneFold(_StratifiedKFold):
    def split(self, X, y, groups=None):
        gen = super().split(X, y, groups)
        try:
            yield next(gen)
        except StopIteration:
            return

import py_files.train_rainfall_v3 as tr3
tr3.StratifiedKFold = OneFold

from py_files.data import preprocessing_v2
from py_files.helpers import add_soil_texture_index
from py_files.train_rainfall_v3 import train_model_rainfall_v3

# ---- paths ----
FILE_PATH = os.path.expanduser(
    "~/Documents/ml-prep/ML-PREP-2025/learn/data/SU_17_training_v3_contri.gpkg"
)
MODEL_SAVE_PATH = str(PROJECT_ROOT / "models" / "verify_k_fix")
TRANSFORMS_DIR  = PROJECT_ROOT / "feature_manifests" / "verify_k_fix"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(TRANSFORMS_DIR, exist_ok=True)

COLUMNS_DROP = [
    'Landslide1', 'descriptio', 'sus_pinn_ground truth', 'ds',
    'cohesion', 'internal_friction', 'sus_pinn_landslide',
    'confusion', 'landslide_preds', 'landslide_probability',
    'Lithology', 'LITHO', 'Geomorphology', 'LITHODESC',
    'LITHO_2', 'LITHODESC_2', 'value',
]
PHYSICS_FEATURES = {
    'Slope_mean', 'BUK_mean', 'PGA2_max',
    'Prc_mean', 'ContributingFactor_mean',
    'SoilThc_mean', 'LULC_majority',
}

# ---- preprocess like notebook cells 6+11 ----
df = gpd.read_file(FILE_PATH)
df, columns, numeric_cols, ind_cols, imp_med = preprocessing_v2(
    df, columns_drop=COLUMNS_DROP, track_imputation=True,
)
df = add_soil_texture_index(df[columns].copy())

with open(PROJECT_ROOT / "feature_manifests" / "v1_cotabato.json") as f:
    manifest = json.load(f)
PGA_COL = manifest["pga_col"]
sel_num = manifest["final_features"]["numerical"]
sel_cat = manifest["final_features"]["categorical"]
sel_num = sel_num + [c for c in ind_cols if c not in sel_num]
sel_feat = sel_num + sel_cat + ["landslide"]

print("\n=========== TRAINING (1 fold, fix applied) ===========")
oof, fold_aucs = train_model_rainfall_v3(
    df,
    sel_num,
    sel_cat,
    sel_feat,
    PGA_COL,
    MODEL_SAVE_PATH,
    epochs=200,
    physics_features=PHYSICS_FEATURES,
    skew_threshold=1.0,
    clip_lower_pct=1,
    clip_upper_pct=99,
    transforms_dir=TRANSFORMS_DIR,
    categorical_encoder='embedding',
    imputed_indicator_cols=ind_cols,
    imputation_medians=imp_med,
)

# ---- dump u_k from the trained fold-1 checkpoint ----
from tensorflow.keras.models import load_model
from py_files.GallenModel_v1 import NewmarkActivation

model = load_model(
    f"{MODEL_SAVE_PATH}/fold-1-model-v3.keras",
    custom_objects={"NewmarkActivation": NewmarkActivation},
)
layer  = model.get_layer("hydraulic_conductivity_v3")
u_k    = layer.u_k.numpy()
k_min  = layer.k_min.numpy()
k_max  = layer.k_max.numpy()
k_now  = k_min + (k_max - k_min) * tf.nn.sigmoid(u_k).numpy()
k_init = k_min + (k_max - k_min) * 0.5

soil_names = ["Sand","Loamy Sand","Sandy Loam","Silt Loam","Loam","Silt",
              "Sandy Clay Loam","Clay Loam","Silty Clay Loam","Sandy Clay",
              "Silty Clay","Clay"]

import pandas as pd
result = pd.DataFrame({
    "soil":      soil_names,
    "u_k":       u_k.round(4),
    "K_init":    k_init.round(3),
    "K_learned": k_now.round(3),
    "moved?":    np.where(np.abs(u_k) > 1e-3, "yes", "no"),
})
print("\n=========== RESULT ===========")
print(result.to_string(index=False))
print(f"\nFold-1 OOF AUC: {fold_aucs[0]:.4f}")
print("\nExpected:")
print("  u_k[7] (Clay Loam) and u_k[11] (Clay) should now be NON-ZERO.")
print("  Other 10 u_k slots stay at 0 (no training samples — different issue).")
