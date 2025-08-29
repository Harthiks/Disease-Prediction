
# Heart Disease Prediction Model

![Machine Learning](https://img.shields.io/badge/Machine-Learning-blue) ![Python](https://img.shields.io/badge/Python-3.x-green) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A robust machine learning project developed in Python to predict the presence of heart disease in patients based on various clinical and demographic features. This repository contains the code for data preprocessing, model training (using Logistic Regression and Random Forest), evaluation, and a pipeline for making new predictions.

## 📁 Dataset

This project uses the **Heart Disease Data** dataset from UCI, sourced from Kaggle.
*   **Source:** [Kaggle: Heart Disease Data (UCI)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
*   The dataset contains medical attributes such as `age`, `sex`, `chol` (cholesterol), `thalch` (maximum heart rate), and many others.
*   The target variable `num` indicates the presence of heart disease.

## 🚀 Features

*   **Data Preprocessing:** Handles missing values (mean imputation for numeric features) and encodes categorical variables.
*   **Exploratory Data Analysis (EDA):** Includes visualizations like histograms and a correlation heatmap to understand data distribution and relationships.
*   **Model Training:** Implements and compares two classification algorithms:
    *   Logistic Regression
    *   Random Forest Classifier
*   **Model Evaluation:** Comprehensive assessment using accuracy, classification reports, and confusion matrices.
*   **Feature Importance:** Identifies and visualizes the most critical features for prediction using Random Forest.
*   **Production Ready:** Includes functionality to save the trained model and scaler for reuse on new data.

## 🛠️ Installation & Usage

### Prerequisites

Ensure you have Python installed along with the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Running the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Harthiks/Disease-Prediction.git
    cd Disease-Prediction
    ```

2.  **Add the Dataset:**
    *   Download the dataset from the Kaggle link above.
    *   Place the `heart_disease_uci.csv` file in the appropriate directory (e.g., `/content/heart-disease/` as in the notebook, or adjust the path in the code).

3.  **Run the Jupyter Notebook:**
    The main logic is contained in `DiseaseP.ipynb`. Open it with Jupyter Notebook or Google Colab and run the cells sequentially.
    ```bash
    jupyter notebook DiseaseP.ipynb
    ```

4.  **Making Predictions on New Data:**
    The code includes a template for loading new patient data (`heart_dataset.csv`), preprocessing it, and making predictions using the saved model.
    *   Format your new data according to the generated `Heart_user_template.csv`.
    *   Use the `joblib` blocks in the notebook to load the model and get predictions.

## 📊 Results

The Random Forest model achieved high performance in predicting heart disease. Key metrics include:
*   **Accuracy:** The model's performance is detailed in the classification report.
*   **Feature Importance:** Top contributing features were identified (e.g., `thalch`, `oldpeak`, `age`).

*(Note: Specific accuracy scores and graphs should be viewed in the notebook output after running the code.)*

## 📂 Repository Structure

```
Disease-Prediction/
├── DiseaseP.ipynb          # Main Jupyter Notebook with full code
├── heart_disease_rf_model.pkl  # Saved Random Forest model (to be generated)
├── model_scalar.pkl        # Saved StandardScaler object (to be generated)
├── Heart_user_template.csv # Template for new prediction input
├── requirements.txt        # Python dependencies (if available)
└── README.md              # This file
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Harthiks/Disease-Prediction/issues) (if you create one).

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙋‍♂️ Author

**Harthik** 
- GitHub: [@Harthiks](https://github.com/Harthiks)

---

**Disclaimer:** This model is for educational and portfolio purposes only and is not intended for direct medical diagnosis. Always consult a healthcare professional for medical advice.