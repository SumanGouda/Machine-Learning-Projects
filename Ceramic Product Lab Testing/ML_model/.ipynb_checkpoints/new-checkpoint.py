# model_pipeline.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

# ----- CONFIG -----
DATA_PATH = "Dataset for model feed - MgO C.xlsx"   # adjust if needed
SHEET_NAME = "synthetic_data"
OUT_DIR = "model_outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
os.makedirs(OUT_DIR, exist_ok=True)

# ----- LOAD -----
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
print("Data shape:", df.shape)

# ----- FEATURES / TARGETS -----
id_cols = ['sample_id', 'setting', 'dominant_carbon_source']
target_cols = [
    'porosity_pct', 'density_g_cm3', 'thermal_conductivity_W_mK',
    'oxidation_mass_loss_pct', 'oxidation_penetration_mm', 'hot_MOR_MPa',
    'slag_contact_angle_deg', 'residual_strength_pct_after_shock'
]
feature_cols = [c for c in df.columns if c not in target_cols + id_cols]

# ----- STRATIFIED SPLIT -----
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                     shuffle=True, stratify=df['dominant_carbon_source'])

X_train = train_df[feature_cols + ['dominant_carbon_source']].copy()
y_train = train_df[target_cols].copy()
X_test = test_df[feature_cols + ['dominant_carbon_source']].copy()
y_test = test_df[target_cols].copy()

# Save test set for your independent verification
test_df.to_csv(os.path.join(OUT_DIR, "test_set_for_verification.csv"), index=False)
test_df.to_excel(os.path.join(OUT_DIR, "test_set_for_verification.xlsx"), index=False)
print("Saved test set copies to", OUT_DIR)

# ----- PREPROCESSING -----
numeric_features = [c for c in feature_cols if df[c].dtype in [np.dtype('float64'), np.dtype('int64')]]
categorical_features = ['dominant_carbon_source']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ----- HELPER: CV EVALUATION (multioutput) -----
def evaluate_model_cv(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    r2s, rmses, maes = [], [], []
    for train_idx, val_idx in kf.split(X):
        Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(Xtr, ytr)
        ypred = pd.DataFrame(model.predict(Xval), index=yval.index, columns=yval.columns)
        r2s.append(np.mean([r2_score(yval[c], ypred[c]) for c in yval.columns]))
        rmses.append(np.mean([np.sqrt(mean_squared_error(yval[c], ypred[c])) for c in yval.columns]))
        maes.append(np.mean([mean_absolute_error(yval[c], ypred[c]) for c in yval.columns]))
    return {'r2_mean': np.mean(r2s), 'rmse_mean': np.mean(rmses), 'mae_mean': np.mean(maes)}

# ----- BASELINE MODELS -----
ridge_pipe = Pipeline([('pre', preprocessor), ('model', MultiOutputRegressor(Ridge(random_state=RANDOM_STATE)))])
rf_pipe = Pipeline([('pre', preprocessor), ('model', MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)))])
gbr_pipe = Pipeline([('pre', preprocessor), ('model', MultiOutputRegressor(GradientBoostingRegressor(random_state=RANDOM_STATE)))])

print("Evaluating Ridge (baseline)...")
print(evaluate_model_cv(ridge_pipe, X_train, y_train, cv=CV_FOLDS))
print("Evaluating RandomForest (baseline)...")
print(evaluate_model_cv(rf_pipe, X_train, y_train, cv=CV_FOLDS))
print("Evaluating GradientBoosting (baseline)...")
print(evaluate_model_cv(gbr_pipe, X_train, y_train, cv=CV_FOLDS))

# ----- LIGHTWEIGHT HYPERPARAMETER SEARCH (RandomizedSearchCV) -----
# Keep n_iter small so it finishes quickly. Increase if you have more compute/time.
param_dist = {
    'model_estimator_n_estimators': [100, 200, 300],
    'model_estimator_max_depth': [None, 6, 12],
    'model_estimator_min_samples_split': [2, 5],
    'model_estimator_min_samples_leaf': [1, 2]
}
rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
rf_mo = MultiOutputRegressor(rf_base)
rf_tune_pipe = Pipeline([('pre', preprocessor), ('model', rf_mo)])

rs = RandomizedSearchCV(rf_tune_pipe, param_distributions=param_dist, n_iter=6, cv=3,
                        scoring='r2', random_state=RANDOM_STATE, n_jobs=1, verbose=1)
print("Running RandomizedSearchCV (quick, n_iter=6)...")
rs.fit(X_train, y_train)
print("Best params:", rs.best_params_)
print("Best CV r2:", rs.best_score_)

best_model = rs.best_estimator_

# ----- FIT BEST MODEL ON FULL TRAINING DATA -----
best_model.fit(X_train, y_train)
joblib.dump(best_model, os.path.join(OUT_DIR, "best_model.joblib"))

# ----- PREDICT ON TEST SET & METRICS -----
y_pred = pd.DataFrame(best_model.predict(X_test), index=y_test.index, columns=y_test.columns)
# Save predictions for verification
pred_df = test_df[['sample_id']].reset_index(drop=True)
pred_df = pd.concat([pred_df, y_test.reset_index(drop=True), y_pred.reset_index(drop=True)], axis=1)
pred_df = pred_df.rename(columns={c: f"pred_{c}" for c in y_test.columns})
pred_df.to_csv(os.path.join(OUT_DIR, "test_set_predictions.csv"), index=False)

# Compute metrics per target
metrics = {}
for col in y_test.columns:
    metrics[col] = {
        'r2': r2_score(y_test[col], y_pred[col]),
        'rmse': np.sqrt(mean_squared_error(y_test[col], y_pred[col])),
        'mae': mean_absolute_error(y_test[col], y_pred[col])
    }
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(os.path.join(OUT_DIR, "test_metrics_per_target.csv"))
print("Test metrics saved to", os.path.join(OUT_DIR, "test_metrics_per_target.csv"))
print(metrics_df)

# ----- PERMUTATION IMPORTANCE -----
perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
# Get transformed feature names
pre = best_model.named_steps['pre']
num_feats = numeric_features
cat_feats = list(pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
feat_names = num_feats + cat_feats
imp_ser = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)
imp_ser.to_csv(os.path.join(OUT_DIR, "permutation_importances.csv"))

# ----- PLOTS -----
sns.set(style='whitegrid', context='notebook')
# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), cmap='coolwarm', center=0)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
plt.close()

# Histograms of targets
for col in y_test.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"hist_{col}.png"))
    plt.close()

# Actual vs Predicted
for col in y_test.columns:
    plt.figure(figsize=(6,6))
    plt.scatter(y_test[col], y_pred[col], alpha=0.7)
    plt.plot([y_test[col].min(), y_test[col].max()], [y_test[col].min(), y_test[col].max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted: {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"actual_vs_pred_{col}.png"))
    plt.close()

# Residual distributions
for col in y_test.columns:
    res = y_test[col] - y_pred[col]
    plt.figure(figsize=(6,4))
    sns.histplot(res, kde=True)
    plt.title(f"Residuals: {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"residuals_{col}.png"))
    plt.close()

# Feature importance bar plot (top 20)
plt.figure(figsize=(8,6))
top20 = imp_ser.head(20)
sns.barplot(x=top20.values, y=top20.index)
plt.title("Permutation Importances (top 20)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importances_top20.png"))
plt.close()

print("All plots and outputs saved in", OUT_DIR)