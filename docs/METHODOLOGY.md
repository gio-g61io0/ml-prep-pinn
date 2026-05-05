# Methodology — Cotabato PINN Landslide Susceptibility Model

This methodology covers two complementary notebooks that share an identical
architecture, preprocessing pipeline, and physics formulation, but differ in
their evaluation protocol:

- **`cotabato_new_slope_unit_v2-8.ipynb`** — 5-fold stratified
  cross-validation; used for unbiased generalization estimates and post-hoc
  diagnostic analysis.
- **`cotabato_production_train.ipynb`** — single 70/30 stratified
  train/validation split; used to produce one deployable model and conduct
  fine-grained physics-pathway diagnostics on a held-out set.

## 1. Study Area and Data

The study area is the North Cotabato province, Philippines, partitioned into
slope units (SU). Each slope unit is the base mapping unit and the unit of
analysis. The dataset (`SU_17_training_v3_contri.gpkg`) records, per slope
unit, a binary co-seismic landslide label (`landslide`) and a suite of
geomorphological, geophysical, hydrological, and lithological covariates
extracted as zonal statistics from raster inputs.

Predictor variables include topographic descriptors (mean slope, elevation,
horizontal/vertical curvature, easting/northing), seismic forcing (PGA),
hydrological forcing (mean monthly precipitation `Prc_mean`,
contributing-area factor, Topographic Wetness Index), soil texture
fractions (clay, silt, sand, in g/kg), bulk unit weight `BUK_mean`, soil
thickness `SoilThc_mean`, vegetation cover (NDVI), land-use category
`LULC_majority`, and Euclidean distances to faults, rivers, and roads.

## 2. Preprocessing

Both notebooks invoke `preprocessing_v2()` from `py_files/data.py`, which:

1. Drops administrative/redundant columns (e.g., prior susceptibility maps,
   lithology duplicates).
2. Filters slope units with mean slope below 10° (mechanically unlikely to
   fail under co-seismic Newmark forcing).
3. Removes rows with any null predictor.
4. Reports per-step row attrition, including positive-class loss, to verify
   that label imbalance is not silently amplified by filtering.

Two additional transformations are applied **only** to non-physics features —
features used directly inside the Mohr–Coulomb / Newmark equations
(`Slope_mean`, `BUK_mean`, `PGA2_max`, `Prc_mean`, `ContributingFactor_mean`,
`SoilThc_mean`, `LULC_majority`) retain their physical units:

5. **Log-transform of right-skewed variables** (`log_transform_skewed`,
   skew threshold |γ| > 1.0) — applied via `log1p` to stabilize the
   distance-to-fault, distance-to-river, distance-to-road, and contributing-
   area features that are otherwise heavy-tailed.
6. **Percentile clipping** at the 1st and 99th percentiles
   (`clip_outliers`) — bounds extreme values without removing rows.

A pairwise correlation check (`check_feature_correlation`,
threshold |r| > 0.9) is run prior to training to flag collinear pairs.

Soil texture index `soil_texture_idx ∈ {0, …, 11}` is derived from the USDA
soil texture triangle (`add_soil_texture_index`) using clay, silt and sand
fractions converted from g/kg to percent. The 12 USDA classes are: Sand,
Loamy Sand, Sandy Loam, Silt Loam, Loam, Silt, Sandy Clay Loam, Clay Loam,
Silty Clay Loam, Sandy Clay, Silty Clay, Clay.

All numeric features except `soil_texture_idx` are then z-score normalized
within the model graph by a per-feature `NormalizationLayer` adapted on the
training fold. The land-cover code `type` is one-hot encoded by a
`CategoricalEncoderLayer`. PGA and `soil_texture_idx` enter the graph
unnormalized because they feed physics layers that require their native
scale.

## 3. Physics-Informed Neural Network (PINN) Architecture

The model `LandslideRainFallV3` (`py_files/LandslideRainfall_v3.py`) is a
hybrid additive-logit network with a residual deep branch and an
auxiliary physics-only output. The forward pass is:

```
encoded features (excluding Prc_mean) ──► Dense(64) ──► 8× [Dense → BN → LeakyReLU(0.2)]
                                                            │
                                                            ▼
                                                Dense(2) → LeakyReLU
                                                            │
                                          ┌─────────────────┴─────────────┐
                                          ▼                               ▼
                                  CohesionLayer (c′)           InternalFrictionLayer (φ′ = σ(·) rad)

soil_texture_idx ──► HydraulicConductivityLayerV3 ──► K (m/hr)
                                                            │
(Prc_mean, ContributingFactor_mean, SoilThc_mean, Slope_mean, K) ──► WetnessLayer ──► m ∈ [0, 0.7]
                                                                                          │
                                                                                          ▼
                              (c′, φ′, slope, PGA, BUK_mean, m) ──► DisplacementLayerRainFall
                                                                                          │
                              ┌────────────────────────┬─────────────────────────────────┤
                              ▼                        ▼                                 ▼
                         FOS (FosLayer)     critical acc. ac (CriticalAcceleration)  D_N (DisplacementIntermediate)
                                                                                          │
                                                                                          ▼
                                                                            NewmarkActivation(threshold=2.0)
                                                                                          │
                                                                                          ▼
                                                                                  physics_prob

(all encoded features) ──► Dense(16) → LeakyReLU → Dense(1) ──► residual logit
                                                                       │
                                          logit(physics_prob) ── Add ──┘
                                                                       │
                                                                       ▼
                                                              sigmoid → final_head
```

### 3.1 Geotechnical sub-network

Eight Dense + BatchNorm + LeakyReLU(α=0.2) layers (widths
[32, 64, 8, 64, 32, 8, 32, 8]) map the **non-rainfall** encoded features to a
2-vector. `CohesionLayer` extracts coordinate 0 via ReLU as cohesion
*c′* (kPa); `InternalFrictionLayer` extracts coordinate 1 through a sigmoid
to produce the internal friction angle *φ′* (rad). Excluding `Prc_mean` from
the geotechnical branch is a deliberate identifiability constraint: it forces
rainfall to influence Factor of Safety *only* through the wetness pathway,
rather than letting the Dense head absorb a rainfall→cohesion shortcut.

### 3.2 Hydrological sub-network

`HydraulicConductivityLayerV3` (`py_files/GallenModel_v3.py`) holds a
12-element learnable vector `u_k`. The hydraulic conductivity for soil class
*i* is

\[
  K_i = K_{\min, i} + (K_{\max, i} - K_{\min, i}) \cdot \sigma(u_{k,i}),
\]

with per-class `(K_min, K_max)` set from USDA literature `Ksat` values
(cm/hr). One-hot lookup selects the per-sample K, then converts to m/hr.

`WetnessLayer` (`py_files/GallenModel_v1.py`) computes the dimensionless
saturation ratio

\[
  m = \frac{R \cdot \alpha}{T \cdot \sin\theta}, \qquad
  R = \mathrm{Prc} \cdot \tfrac{0.001}{720}, \qquad
  T = K \cdot d_s,
\]

where *R* is rainfall in m/hr (converted from mm/month assuming a 30-day
month), α is the contributing-area factor, *T* is transmissivity, *d*₍s₎
is soil thickness, and θ is slope. The result is clipped to [0, 1] inside
the layer and then passed through a sigmoid activation
(`Activation('sigmoid', name="m_clip")`) before entering the displacement
layer.

### 3.3 Newmark physics

`DisplacementLayerRainFall` implements the limit-equilibrium Factor of
Safety with pore-pressure correction:

\[
  \mathrm{FOS} =
    \frac{c'}{\gamma_b\, d_s\, \sin\theta} +
    \frac{\tan\varphi'}{\tan\theta} -
    \frac{m\,\gamma_w \tan\varphi'}{\gamma_b \tan\theta},
\]

with *d*₍s₎ = 3.33 m fixed slab thickness, γ_w = 9.81 N/m³ unit weight of
water, and γ_b derived from `BUK_mean` (kN/m³ → N/m³). Cohesion is converted
kPa→Pa.

The critical seismic acceleration is

\[
  a_c = (\mathrm{FOS} - 1) \cdot g \cdot \sin\theta,
\]

passed through ReLU so that stable slopes (FOS < 1) yield zero. The ratio
*a*₍c₎/(g·PGA) is clipped to [0.001, 0.75] to keep the Jibson regression in
its calibrated domain. Newmark cumulative displacement is then

\[
  \log_{10} D_N = 0.215 + \log_{10}\!\left[(1 - a_c/a_{\max})^{2.341} (a_c/a_{\max})^{-1.438}\right] + 0.51.
\]

Susceptibility is mapped from displacement by a fixed sigmoid threshold
(`NewmarkActivation`, threshold = 2.0):

\[
  P_{\text{phys}} = \frac{1}{1 + \exp(2.0 - D_N)}.
\]

### 3.4 Hybrid additive-logit head

The residual deep branch `Dense(16, LeakyReLU) → Dense(1)` consumes the
**full** encoded feature vector (including `Prc_mean`) and produces a scalar
logit correction *ρ*. The final landslide probability is

\[
  P_{\text{final}} = \sigma\bigl(\mathrm{logit}(P_{\text{phys}}) + \rho\bigr).
\]

A numerically-stabilized `LogitLayer` (clipping at 1e-6) inverts the physics
sigmoid before addition. The physics probability is exposed as a separate
output `physics_prob` so that auxiliary supervision can prevent the residual
branch from dominating and the physics layers from collapsing.

## 4. Loss, Optimization, and Class Imbalance

Training uses Adam (η = 1e-4) and a multi-output loss:

- `final_head`: Dice-Cross-Entropy (`DiceCrossEntropyLoss`), weight 1.0.
- `physics_prob`: Binary Cross-Entropy, weight `aux_weight = 0.3`.

Because the slope-unit landslide rate is highly imbalanced, per-sample
weights {0: 1, 1: 5} are applied via `to_multi_output_ds`
(class-weight kwargs are not supported on multi-output Keras models, so the
weights are baked into the dataset). Monitored metrics include AUC, binary
IoU, and accuracy on the final head, and AUC on the physics-only head.

Training callbacks: early stopping on `val_final_head_auc` (mode = max,
patience = 5, restore best weights), and `ModelCheckpoint` saving the best
epoch per fold.

Reproducibility is enforced by `set_seed(42)` (Python `random`, NumPy,
TensorFlow, `PYTHONHASHSEED`, and `TF_DETERMINISTIC_OPS=1`) at the top of
every notebook.

## 5. Evaluation Protocol

### 5.1 Cross-validated evaluation (`cotabato_new_slope_unit_v2-8.ipynb`)

Five-fold stratified k-fold cross-validation is run on the preprocessed
dataset, stratifying on `landslide`. For each fold, an independent model is
fit, the best epoch checkpoint is saved as
`fold-{k}-model-v3.keras`, and held-out predictions are accumulated into a
full out-of-fold prediction vector. Per-fold ROC curves are recorded with
mean ± inter-fold variability.

The fold with the highest validation AUC (fold 3 in the analyses) is loaded
as the canonical model for the post-training diagnostic suite. Out-of-fold
predictions on the **full** preprocessed dataset (without outlier clipping
on the validation pass) form the basis of all subsequent metrics. Standard
discrimination metrics include ROC-AUC, balanced accuracy, confusion
matrix, calibration plot with Brier score, precision–recall curve and area
(AUPR), success-rate curves of the cumulative landslide-area capture, and
landslide density per susceptibility class.

### 5.2 Production model evaluation (`cotabato_production_train.ipynb`)

A single stratified 70/30 train/validation split (random_state=42) is used
to fit one deployable model with the same architecture and the same
loss/metrics/callbacks. Validation discrimination metrics, calibration,
precision–recall, and full-dataset susceptibility maps are reported.

## 6. Post-training Diagnostics

The cross-validated notebook (`v2-8`) emphasizes ensemble and inventory
diagnostics, while the production notebook focuses on physics-pathway
diagnostics. Diagnostic methods are summarized below.

### 6.1 Intermediate physics extraction

Sub-models are constructed by re-routing
`tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("…").output)`
for each named physics layer (`fos_layer`, `cohesion_layer`,
`internal_friction`, `m_clip`, `displacement_layer`). This yields per-pixel
estimates of FOS, *c′*, *φ′*, *m*, and *D*₍N₎ that are then analyzed
separately from the susceptibility output.

### 6.2 Cross-fold stability and ensemble uncertainty (v2-8)

All five fold checkpoints are loaded; per-fold susceptibility, *c′*, *φ′*,
and FOS are stacked. Distributional stability is summarized with
fold-overlay KDEs (`plot_fold_stability`). Ensemble mean and standard
deviation (`fold_ensemble_uncertainty`) are mapped spatially; pixels with
high ensemble mean and low standard deviation are reported as confident
high-susceptibility predictions, while high standard deviation indicates
epistemic uncertainty driven by feature-space gaps in the inventory.

### 6.3 SHAP-based feature attribution (v2-8)

Tree-agnostic SHAP values are computed via the project helper
`compute_shap_values` (200 background samples, 500 explanation samples).
Mean |SHAP| ranks features by global importance; beeswarm and bar plots
display directionality.

### 6.4 Geotechnical parameters versus literature (v2-8)

Predicted *c′* and *φ′* are grouped by `soil_texture` label and plotted as
boxplots versus the published USDA per-class ranges encoded in
`SoilConditionedGeotechLayerV3.COH_RANGES` and `IFI_RANGES`. The learned
hydraulic conductivity (`u_k`) is similarly back-transformed into cm/h and
tabulated against literature `Ksat` bounds.

### 6.5 Incomplete-inventory false-positive analysis (v2-8)

The Cotabato inventory captures landslides from a single earthquake event.
Slope units labelled "no landslide" may nonetheless be genuinely
susceptible. To test whether the model's apparent false positives are
**physically meaningful generalizations** rather than overpredictions:

1. Predictions are partitioned into TP/FP/TN/FN at threshold 0.5
   (`classify_predictions`).
2. KDE distributions of geomorphological features (slope, elevation,
   precipitation, TWI, soil thickness, BUK, contributing area, distance to
   faults/rivers/roads, NDVI) are compared across groups
   (`plot_fp_vs_tp_distributions`).
3. Mann–Whitney U tests (`fp_tp_statistical_tests`) test the dual hypotheses
   that (a) FP and TP come from the same distribution (p > 0.05 expected if
   FPs are physically landslide-like) and (b) FP and TN come from different
   distributions (p < 0.05 expected if FPs are not random noise).
4. Spatial coherence is assessed by mapping TP/FP/TN/FN polygons
   (`plot_fp_map`); FP clusters adjacent to TP clusters support the
   "missed inventory" interpretation.
5. Physics-intermediate KDEs (FOS, *D*₍N₎, *c′*, *φ′*, *m*) are compared
   across TP / FP / TN to confirm geotechnical reasoning is consistent
   between known and putatively missed landslides.

### 6.6 Physics-pathway diagnostics (production)

To verify that each physics sub-pathway is functional and to localize
prediction error sources, the production notebook decomposes the validation
set into TP / FN / FP / TN and inspects upstream physics quantities:

- **Displacement distribution by outcome** — log-scale histograms split by
  TP/FN/FP/TN, with the activation threshold of 2.0 m overlaid. The subset
  `(y_true=1) ∧ (D_N < 2.0)` is the "Newmark-underestimated" population that
  the deterministic physics chain misses entirely.
- **FOS by outcome** — histograms and FOS-vs-slope scatter plots reveal
  whether FN pixels are physically considered stable (FOS ≫ 1) or whether
  the threshold is the binding limitation.
- **Saturation ratio *m* by outcome** — diagnoses whether the rainfall →
  wetness pathway is firing. If FN *m* is collapsed near zero despite high
  `Prc_mean`, the coupling is broken; if FN *m* matches TP *m*, the issue
  is downstream.
- **FOS-vs-rainfall and FOS-vs-*m*** — Spearman ρ split by outcome
  quantifies whether the pore-pressure pathway has gradient. Negative ρ
  indicates an active pathway; flat ρ indicates the model has learned to
  ignore the rainfall→FOS link.
- **Hybrid vs physics-only predictions** — scatter of `final_head` against
  `physics_prob`, distribution of the residual correction
  `δ = final_head - physics_prob` per outcome, and FN-recovery counts
  (landslides physics-only misses but the hybrid rescues, and vice versa)
  quantify the residual branch's net contribution.
- **Spatial distribution of FNs** — DBSCAN clustering on FN polygon
  centroids (eps = 500 m, min_samples = 3) separates "clustered" from
  "isolated" FNs; clustered FNs are compared against their nearest TP
  within 2 km via Kolmogorov–Smirnov tests on each input feature, while
  isolated FNs are compared against the global TP distribution. Clustered
  FNs distributionally identical to nearby TPs are interpreted as inventory
  noise; statistically distinct clusters indicate hidden local factors not
  captured by the present feature set.
- **FN vs TP feature distributions** — KS tests on
  Slope/PGA/Prc/BUK/SoilThc/ContributingArea/TWI/Elev/NDVI/Clay/Sand/Silt
  diagnose trigger-type mismatches (e.g., FN cluster at lower slope and
  higher rainfall would indicate a rainfall-induced subset that the
  co-seismic Newmark formulation underweights).

## 7. Implementation Notes

- **Reproducibility**: deterministic ops, fixed seed 42, identical
  preprocessing across both notebooks.
- **Custom layer serialization**: every physics layer carries
  `@tf.keras.utils.register_keras_serializable()` so checkpoints round-trip
  cleanly via `tf.keras.models.load_model(...)` with
  `custom_objects={"NewmarkActivation": NewmarkActivation}`.
- **Validation purity**: outlier clipping and log transforms are applied
  before the train/test partition in both notebooks. The cross-validated
  notebook re-loads the raw file for diagnostic prediction without
  re-clipping, so validation feature distributions are not biased by
  training-set quantile bounds.
- **Physics-feature fidelity**: `Slope_mean`, `BUK_mean`, `PGA2_max`,
  `Prc_mean`, `ContributingFactor_mean`, `SoilThc_mean`, and
  `LULC_majority` are excluded from log transforms and outlier clipping so
  that the values entering Mohr–Coulomb / Newmark equations remain in their
  real physical units.
