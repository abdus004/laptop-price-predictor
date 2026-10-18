
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("laptops_cleaned.csv")

# Step 2: Select features for our first model
features = [
    "brand",
    "device_category",
    "cpu_brand",
    "cpu_family",
    "cpu_series",
    "cpu_model",
    "cpu_core_count",
    "cpu_thread_count",
    "ram_gb",
    "storage_gb",
    "display_width_px",
    "display_height_px",
    "display_size_inch",
    "gpu_brand",
    "gpu_series",
    "gpu_model",
    "gpu_type",
    "os_name",
    "warranty_years",
]

target = "price"

# Step 3: Create our modelling dataset
model_df = df[features + [target]].copy()

# Step 4: Check the selected data
print("Original dataset shape:", df.shape)
print("Selected dataset shape:", model_df.shape)

print("\nMissing values in selected features:")
print(model_df[features].isnull().sum().to_string())

print("\nTarget summary:")
print(model_df[target].describe())

print("\nFirst five rows:")
print(model_df.head().to_string(index=False))

print("\nFeature preparation check complete.")
