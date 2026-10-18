
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Load resources ----------

@st.cache_resource
def load_model():
    return joblib.load("extra_trees_laptop_price_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("laptops_cleaned.csv")


model = load_model()
df = load_data()


def options(column, include_blank=False):
    values = sorted(
        df[column].dropna().unique().tolist(),
        key=lambda value: str(value).lower(),
    )
    if include_blank:
        return ["Not specified"] + values
    return values


def numeric_options(column, include_blank=False):
    values = sorted(df[column].dropna().unique().tolist())
    if include_blank:
        return ["Not specified"] + values
    return values




# ---------- Page styling ----------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #080F1D, #0D1729 55%, #101C32);
        color: #E2E8F0;
    }

    .stApp [data-testid="stHeader"] {
        background: rgba(8, 15, 29, 0.95);
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Headings and text */
    .stApp h1, .stApp h2, .stApp h3,
    .stApp p, .stApp label,
    .stApp [data-testid="stMarkdownContainer"] {
        color: #E2E8F0;
    }

    .stApp [data-testid="stCaptionContainer"] {
        color: #94A3B8;
    }

    /* Main form card */
    div[data-testid="stForm"] {
        background: #111D30;
        padding: 1.5rem;
        border: 1px solid #263650;
        border-radius: 16px;
    }

    /* Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #172338;
        border-color: #33445E;
    }

    div[data-baseweb="select"] * {
        color: #E2E8F0;
    }

    /* Dropdown menu */
    ul[role="listbox"] {
        background-color: #172338;
    }

    li[role="option"] {
        background-color: #172338;
        color: #E2E8F0;
    }

    li[role="option"]:hover {
        background-color: #263B5A;
    }

    /* Advanced specifications */
    div[data-testid="stExpander"] {
        background: #111D30;
        border: 1px solid #263650;
        border-radius: 12px;
    }

    /* Prediction button */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #2563EB, #3B82F6);
        color: #FFFFFF;
        border: 1px solid #3B82F6;
        border-radius: 9px;
        min-height: 46px;
        font-weight: 600;
    }

    div[data-testid="stFormSubmitButton"] button p {
        color: #FFFFFF;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #1D4ED8;
        border-color: #60A5FA;
    }

    /* Prediction result */
    div[data-testid="stMetric"] {
        background: #111D30;
        border: 1px solid #263650;
        padding: 1.25rem;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
    }

    div[data-testid="stMetricValue"] {
        color: #60A5FA;
    }

    /* Divider */
    hr {
        border-color: #263650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Header ----------

st.title("Laptop Price Predictor")
st.write(
    "Configure a laptop and get an estimated price "
    "from a machine-learning model."
)

st.caption(
    "Estimates are based on historical dataset listings. "
    "They are not guaranteed current market prices."
)

st.divider()

# ---------- Input form ----------

with st.form("laptop_form"):

    st.subheader("1. Basic Specifications")
    st.write("Choose the details you know.")

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Laptop Brand", options("brand"))
        category = st.selectbox(
            "Laptop Category", options("device_category")
        )
        cpu_brand = st.selectbox("Processor Brand", options("cpu_brand"))
        cpu_family = st.selectbox(
            "Processor Family", options("cpu_family")
        )
        ram = st.selectbox(
            "RAM (GB)", numeric_options("ram_gb")
        )

    with col2:
        storage = st.selectbox(
            "Storage (GB)", numeric_options("storage_gb")
        )
        os_name = st.selectbox(
            "Operating System", options("os_name")
        )
        gpu_type = st.selectbox("Graphics Type", options("gpu_type"))
        display_size = st.selectbox(
            "Display Size (inches)",
            numeric_options("display_size_inch"),
        )

    with st.expander("Advanced Specifications (Optional)"):

        st.write(
            "Leave fields as Not specified if you don't know "
            "the exact hardware details."
        )

        col3, col4 = st.columns(2)

        with col3:
            cpu_series = st.selectbox(
                "Processor Series",
                options("cpu_series", include_blank=True),
            )
            cpu_model = st.selectbox(
                "Exact CPU Model",
                options("cpu_model", include_blank=True),
            )
            cpu_cores = st.selectbox(
                "CPU Cores",
                numeric_options("cpu_core_count", include_blank=True),
            )
            cpu_threads = st.selectbox(
                "CPU Threads",
                numeric_options("cpu_thread_count", include_blank=True),
            )
            display_width = st.selectbox(
                "Display Width (pixels)",
                numeric_options("display_width_px", include_blank=True),
            )

        with col4:
            gpu_brand = st.selectbox(
                "GPU Brand",
                options("gpu_brand", include_blank=True),
            )
            gpu_series = st.selectbox(
                "GPU Series",
                options("gpu_series", include_blank=True),
            )
            gpu_model = st.selectbox(
                "Exact GPU Model",
                options("gpu_model", include_blank=True),
            )
            display_height = st.selectbox(
                "Display Height (pixels)",
                numeric_options("display_height_px", include_blank=True),
            )
            warranty = st.selectbox(
                "Warranty (years)",
                numeric_options("warranty_years", include_blank=True),
            )

    submitted = st.form_submit_button(
        "Predict Laptop Price", use_container_width=True
    )

# ---------- Prediction ----------

if submitted:

    def selected_value(value):
        return None if value == "Not specified" else value

    input_data = pd.DataFrame([{
        "brand": brand,
        "device_category": category,
        "cpu_brand": cpu_brand,
        "cpu_family": cpu_family,
        "cpu_series": selected_value(cpu_series),
        "cpu_model": selected_value(cpu_model),
        "gpu_brand": selected_value(gpu_brand),
        "gpu_series": selected_value(gpu_series),
        "gpu_model": selected_value(gpu_model),
        "gpu_type": gpu_type,
        "os_name": os_name,
        "cpu_core_count": selected_value(cpu_cores),
        "cpu_thread_count": selected_value(cpu_threads),
        "ram_gb": ram,
        "storage_gb": storage,
        "display_width_px": selected_value(display_width),
        "display_height_px": selected_value(display_height),
        "display_size_inch": display_size,
        "warranty_years": selected_value(warranty),
    }])

    try:
        predicted_price = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Estimated Laptop Price")
        st.metric(
            label="Predicted price",
            value=f"₹{predicted_price:,.0f}",
        )
        st.caption(
            "This is a model estimate, not a live retail quotation."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Please check the selected specifications. "
            "We will investigate any model or input compatibility issue."
        )
