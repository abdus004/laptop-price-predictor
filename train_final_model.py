
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor


# 1. Load the dataset
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

df = df[features + ["price"]].dropna(subset=["price"])

X = df[features]
y = df["price"]

# 2. Prepare categorical and numerical inputs
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_features),
    ("numerical", numerical_pipeline, numerical_features),
])

# 3. Build the selected model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )),
])

# 4. Train on all available labelled rows for deployment
print(f"Training Extra Trees on {len(df)} laptop records...")

model.fit(X, y)

# 5. Save as a NEW model file
output_file = "extra_trees_laptop_price_model.joblib"
joblib.dump(model, output_file)

print("\nFinal model trained successfully.")
print(f"Saved model: {output_file}")
print("Existing model files were not modified.")
