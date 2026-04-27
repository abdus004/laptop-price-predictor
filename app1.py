import streamlit as st
import pickle
import pandas as pd

# load model
model = pickle.load(open("model.pkl", "rb"))

st.title("💻 Laptop Price Predictor (INR)")
st.write("Enter laptop specifications:")

# -------- INPUTS -------- #

company = st.selectbox("Company", ["Dell", "HP", "Apple", "Lenovo", "Asus", "Acer"])
typename = st.selectbox("Type", ["Notebook", "Ultrabook", "Gaming"])
cpu = st.selectbox("CPU", ["i3", "i5", "i7"])

ram = st.selectbox("RAM (GB)", [4, 8, 16, 32])
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])
hdd = st.selectbox("HDD (GB)", [0, 500, 1000, 2000])

weight = st.slider("Weight (kg)", 1.0, 3.5, 1.5)

# -------- PREDICT -------- #

if st.button("Predict Price"):

    input_dict = {
        "Ram": ram,
        "Weight": weight,
        "SSD": ssd,
        "HDD": hdd
    }

    input_df = pd.DataFrame([input_dict])

    # create all required columns
    for col in model.feature_names_in_:

        if col.startswith("Company_"):
            input_df[col] = 1 if col == f"Company_{company}" else 0

        elif col.startswith("TypeName_"):
            input_df[col] = 1 if col == f"TypeName_{typename}" else 0

        elif col.startswith("Cpu_"):
            input_df[col] = 1 if col == f"Cpu_{cpu}" else 0

        elif col.startswith("OpSys_"):
            input_df[col] = 0

    # match exact structure
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)

    st.success(f"💰 Estimated Price: ₹{int(prediction[0])}")