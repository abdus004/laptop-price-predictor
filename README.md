# Laptop Price Predictor

A machine learning web application that predicts laptop prices in Indian Rupees (₹) based on hardware specifications. Built using Python, Scikit-learn, and Streamlit.

**GitHub Repository:** [abdus004/laptop-price-predictor](https://github.com/abdus004/laptop-price-predictor)

## Project Overview

The Laptop Price Predictor estimates laptop prices based on specifications such as brand, processor, RAM, storage, operating system, graphics, and display size.

The application uses an **Extra Trees Regressor** to learn patterns from historical laptop listing data and predict an estimated price for user-selected specifications.

## Features

- Predict laptop prices in Indian Rupees (₹).
- Interactive web interface using Streamlit.
- Basic specifications for quick predictions.
- Optional advanced specifications for more detailed inputs.
- Data preprocessing for missing values and categorical features.
- Machine learning model saved using Joblib.
- User-friendly interface with a dark-themed design.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Model:** Extra Trees Regressor
- **Data Processing:** Pandas
- **Web Framework:** Streamlit
- **Model Serialization:** Joblib

## Dataset

The project uses a cleaned laptop specifications dataset containing **992 records and 27 columns**, including the target variable `price`.

**Dataset source:** [Laptop Specs Dataset](https://github.com/abhinavflac/laptops-specs-dataset)

The dataset contains historical listing prices. Therefore, predictions are estimates and may differ from current retail prices.

## Machine Learning Workflow

1. Load and inspect the laptop dataset.
2. Prepare the input features and target price.
3. Handle missing numerical and categorical values.
4. Encode categorical features using One-Hot Encoding.
5. Train and evaluate multiple regression algorithms.
6. Select Extra Trees based on the evaluation results.
7. Retrain the selected model on all 992 records.
8. Save the trained model using Joblib.
9. Load the model in Streamlit to predict prices from user inputs.

## Model Evaluation

The selected model was evaluated using a held-out test set before the final model was retrained on the complete dataset.

| Evaluation Metric | Result |
|---|---:|
| Algorithm | Extra Trees Regressor |
| Training records during evaluation | 793 |
| Test records | 199 |
| Mean Absolute Error (MAE) | ₹11,448.19 |
| Root Mean Squared Error (RMSE) | ₹19,307.03 |
| R² Score | 0.8915 |

### Understanding the Results

- **MAE:** Predictions differed from actual prices by approximately ₹11,448 on average in absolute terms on the held-out test set.
- **RMSE:** Larger prediction errors have a greater effect on this metric. The test RMSE was approximately ₹19,307.
- **R² Score:** The model achieved an R² score of 0.8915 on the held-out test set, indicating that it explained a substantial proportion of the variation in test prices.

**Evaluation note:** These metrics belong to the held-out evaluation model. The final saved model was subsequently trained on all 992 records, so these figures should not be described as a separate test of that final model.

## Installation and Setup

### Prerequisites

- Python installed on your computer.
- Git installed on your computer.
- Basic knowledge of running commands in a terminal.

### 1. Clone the repository

```bash
git clone https://github.com/abdus004/laptop-price-predictor.git
cd laptop-price-predictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## Project Structure

```text
laptop-price-predictor/
│
├── app.py
├── train_final_model.py
├── extra_trees_laptop_price_model.joblib
├── laptops_cleaned.csv
├── requirements.txt
└── README.md
```

Additional experiment scripts and evaluation result files may also be present in the repository.

## Limitations

- Prediction accuracy depends on the quality and coverage of the training dataset.
- Historical prices may not reflect current market prices.
- Discounts, availability, seller differences, and market changes can affect actual prices.
- Leaving optional specifications unspecified may affect individual predictions.
- Predictions are estimates, not guaranteed selling or purchasing prices.

## Future Improvements

- Update the dataset with newer laptop listings.
- Evaluate prediction errors across different brands and price ranges.
- Deploy the application online.
- Add visualizations of model performance.
- Improve price explanations and prediction transparency.

## Author

**Abdus Salaam A.**

AI & Data Science Student

GitHub: [@abdus004](https://github.com/abdus004)

## Acknowledgements

- Dataset contributors for providing laptop specifications data.
- The Python, Pandas, Scikit-learn, Joblib, and Streamlit communities for their tools and documentation.
