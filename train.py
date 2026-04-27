import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# load dataset
df = pd.read_csv("laptop_data.csv")

# clean column names
df.columns = df.columns.str.strip()

# drop useless column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# ---------------- CLEANING ---------------- #

# CPU → extract i3/i5/i7
df["Cpu"] = df["Cpu"].str.extract("(i3|i5|i7)")

# RAM → "8GB" → 8
df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)

# Weight → "1.37kg" → 1.37
df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

# MEMORY → extract SSD & HDD
df["SSD"] = df["Memory"].str.extract(r"(\d+)GB SSD")
df["HDD"] = df["Memory"].str.extract(r"(\d+)TB HDD")

df["HDD"] = df["HDD"].astype(float) * 1000
df["SSD"] = df["SSD"].fillna(0).astype(int)
df["HDD"] = df["HDD"].fillna(0)

# drop unused columns
df.drop(["Memory", "Gpu", "ScreenResolution"], axis=1, inplace=True)

# ---------------- ENCODING ---------------- #

df = pd.get_dummies(df, columns=["Company", "TypeName", "Cpu", "OpSys"], drop_first=True)

# ---------------- MODEL ---------------- #

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved as model.pkl")