
import time
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 55)
print("LAPTOP PRICE PREDICTOR - MODEL EXPERIMENT")
print("=" * 55)

print("\nLoading dataset...")

df = pd.read_csv("laptops_cleaned.csv")

target = "price"

categorical_features = [
    "brand",
    "device_category",
    "cpu_brand",
    "cpu_family",
    "cpu_series",
    "cpu_model",
    "gpu_brand",
    "gpu_series",
    "gpu_model",
    "gpu_type",
    "os_name"
]

numerical_features = [
    "cpu_core_count",
    "cpu_thread_count",
    "ram_gb",
    "storage_gb",
    "display_width_px",
    "display_height_px",
    "display_size_inch",
    "warranty_years"
]

features = categorical_features + numerical_features

# Select only the columns required for this experiment.
df = df[features + [target]].copy()

# Remove rows where the target price is missing.
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

print(f"Dataset rows: {len(df)}")
print(f"Number of input features: {len(features)}")


# ============================================================
# 2. CREATE A FRESH TRAIN/TEST SPLIT
# ============================================================

print("\nCreating training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=2026
)

print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")


# ============================================================
# 3. PREPROCESSING
# ============================================================

# Missing categorical values are filled with the most common value.
# Unknown categories are safely handled by OneHotEncoder.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Missing numerical values are filled with the training-fold median.
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_features),
    ("numerical", numerical_pipeline, numerical_features)
])


# ============================================================
# 4. DEFINE FOUR MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# ============================================================
# 5. CROSS-VALIDATION ON TRAINING DATA ONLY
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results_list = []

print("\n" + "=" * 55)
print("STARTING 5-FOLD CROSS-VALIDATION")
print("=" * 55)

for name, model in models.items():

    print(f"\nTesting: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    start_time = time.time()

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1
    )

    elapsed_time = time.time() - start_time

    fold_maes = -scores
    mean_mae = fold_maes.mean()
    std_mae = fold_maes.std()

    cv_results_list.append({
        "Model": name,
        "Mean CV MAE (INR)": round(mean_mae, 2),
        "MAE Std (INR)": round(std_mae, 2),
        "CV Time (seconds)": round(elapsed_time, 2)
    })

    print(f"Mean CV MAE: Rs. {mean_mae:,.2f}")
    print(f"MAE standard deviation: Rs. {std_mae:,.2f}")
    print(f"Time taken: {elapsed_time:.2f} seconds")


# Sort models from lowest to highest cross-validation MAE.
cv_results = pd.DataFrame(cv_results_list)

cv_results = cv_results.sort_values(
    by="Mean CV MAE (INR)",
    ascending=True
).reset_index(drop=True)

cv_results.to_csv(
    "baseline_cv_results.csv",
    index=False
)


# ============================================================
# 6. SELECT THE MODEL USING CROSS-VALIDATION
# ============================================================

best_name = cv_results.iloc[0]["Model"]
best_model = models[best_name]

print("\n" + "=" * 55)
print("MODEL SELECTED USING CROSS-VALIDATION")
print("=" * 55)

print(f"Selected model: {best_name}")
print(
    "Mean CV MAE: Rs. "
    f"{cv_results.iloc[0]['Mean CV MAE (INR)']:,.2f}"
)


# ============================================================
# 7. TRAIN SELECTED MODEL AND EVALUATE ON TEST DATA
# ============================================================

print("\nTraining the selected model...")

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

final_pipeline.fit(X_train, y_train)

print("Evaluating on the held-out test set...")

predictions = final_pipeline.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    predictions
)

test_rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

test_r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# 8. SAVE FINAL TEST RESULTS
# ============================================================

final_results = pd.DataFrame([{
    "Selected Model": best_name,
    "Test MAE (INR)": round(test_mae, 2),
    "Test RMSE (INR)": round(test_rmse, 2),
    "Test R2": round(test_r2, 4),
    "Training Rows": len(X_train),
    "Test Rows": len(X_test)
}])

final_results.to_csv(
    "baseline_final_test_results.csv",
    index=False
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 55)
print("CROSS-VALIDATION COMPARISON")
print("=" * 55)

print(cv_results.to_string(index=False))

print("\n" + "=" * 55)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 55)

print(final_results.to_string(index=False))

print("\n" + "=" * 55)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 55)

print("\nFiles created:")
print("1. baseline_cv_results.csv")
print("2. baseline_final_test_results.csv")

print("\nExisting .joblib model files were not modified.")
print("Your app.py was not modified.")
