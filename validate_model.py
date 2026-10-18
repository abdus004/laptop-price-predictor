
# ============================================
# INDIAN LAPTOP PRICE PREDICTOR
# Phase 5: Cross-Validation and Model Validation
# ============================================

import time
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 1: LOAD DATA
print("=" * 65)
print("       LAPTOP PRICE PREDICTION - MODEL VALIDATION")
print("=" * 65)

df = pd.read_csv("laptops_cleaned.csv")

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
    "os_name",
]

numerical_features = [
    "cpu_core_count",
    "cpu_thread_count",
    "ram_gb",
    "storage_gb",
    "display_width_px",
    "display_height_px",
    "display_size_inch",
    "warranty_years",
]

features = categorical_features + numerical_features

X = df[features]
y = df["price"]

# Keep the same split for comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Test records:", len(X_test))


# STEP 2: CREATE A FRESH PREPROCESSOR
def create_preprocessor():

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ])


# STEP 3: DEFINE MODELS
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# STEP 4: FIVE-FOLD CROSS-VALIDATION
# Only training data is used in this stage.

cv_results = []

for name, model in models.items():

    print("\n" + "-" * 65)
    print("Validating:", name)

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", model)
    ])

    start = time.time()

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        },
        n_jobs=1,
        error_score="raise"
    )

    elapsed = time.time() - start

    # Scikit-learn returns negative MAE and RMSE
    # because larger scoring values are preferred.
    fold_mae = -scores["test_mae"]
    fold_rmse = -scores["test_rmse"]
    fold_r2 = scores["test_r2"]

    result = {
        "Model": name,
        "Mean CV MAE (INR)": fold_mae.mean(),
        "MAE Std (INR)": fold_mae.std(),
        "Mean CV RMSE (INR)": fold_rmse.mean(),
        "Mean CV R2": fold_r2.mean(),
        "R2 Std": fold_r2.std(),
        "CV Time (seconds)": elapsed
    }

    cv_results.append(result)

    print("Fold MAE values:", fold_mae.round(2))
    print("Mean CV MAE:", round(fold_mae.mean(), 2))
    print("MAE standard deviation:", round(fold_mae.std(), 2))
    print("Mean CV RMSE:", round(fold_rmse.mean(), 2))
    print("Mean CV R2:", round(fold_r2.mean(), 4))
    print("Validation time:", round(elapsed, 2), "seconds")


# STEP 5: COMPARE CROSS-VALIDATION RESULTS
results_df = pd.DataFrame(cv_results)

results_df = results_df.sort_values(
    by="Mean CV MAE (INR)"
).reset_index(drop=True)

print("\n" + "=" * 90)
print("                 CROSS-VALIDATION COMPARISON")
print("=" * 90)

print(results_df.round(4).to_string(index=False))

results_df.to_csv("cross_validation_results.csv", index=False)

best_name = results_df.iloc[0]["Model"]
print("\nBest model by mean CV MAE:", best_name)


# STEP 6: FIT SELECTED MODEL ON ALL TRAINING DATA
# The test set is not used for model selection.

best_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("model", models[best_name])
])

print("\nTraining selected model on all training records...")
best_pipeline.fit(X_train, y_train)

test_predictions = best_pipeline.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = mean_squared_error(
    y_test, test_predictions
) ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("\n" + "=" * 65)
print("              HELD-OUT TEST RESULTS")
print("=" * 65)

print("Selected model:", best_name)
print("Test MAE:", round(test_mae, 2), "rupees")
print("Test RMSE:", round(test_rmse, 2), "rupees")
print("Test R2:", round(test_r2, 4))


# STEP 7: SAVE SELECTED MODEL
import joblib

joblib.dump(best_pipeline, "best_laptop_price_model.joblib")

print("\nCross-validation results saved to:")
print("cross_validation_results.csv")

print("\nSelected model saved to:")
print("best_laptop_price_model.joblib")

print("\nModel validation completed!")
