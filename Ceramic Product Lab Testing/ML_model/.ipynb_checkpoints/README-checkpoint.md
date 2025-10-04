# MgO-C Refractory Properties Prediction Model

## 📝 Project Overview

This project contains a machine learning pipeline designed to predict the key performance properties of Magnesia-Carbon (MgO-C) refractory materials. By providing the material's **raw ingredient composition** and its **processing parameters**, the model can accurately forecast 8 critical properties, such as porosity, density, and thermal shock resistance.

The goal is to accelerate materials science research by allowing for rapid virtual experimentation, reducing the time and cost associated with physical sample creation and testing.

---

## 📁 File Structure

.
├── Dataset for model feed - MgO C.xlsx    (The full dataset used for training)
├── model_pipeline.ipynb                 (Script to train the model from scratch)
├── run_prediction.ipynb                 (Script to use the trained model for new predictions)
├── prediction_results.csv            (Example output file created by run_prediction.py)
├── model_outputs/                    (Folder containing all outputs from the training script)
│   ├── best_model.joblib               (The final, trained model file)
│   ├── test_set_for_verification.csv   (The 20% of data used for final testing)
│   ├── test_metrics_per_target.csv     (Performance scores of the model)
│   ├── permutation_importances.csv     (List of the most influential features)
│   ├── feature_importances_top20.png   (Plot of the most influential features)
│   ├── learning_curves/                (Directory with plots showing how the model learned)
│   └── residual_plots/                 (Directory with diagnostic plots of model errors)
└── README.md                         (This file)


---

## ⚙️ How the Model Was Built (`model_pipeline.py`)

The model was built using a systematic process that ensures robust and reliable performance. The entire process is encapsulated in the **`model_pipeline.ipynb`** script.

1.  **Data Loading & Standardization**: The script begins by loading the main data file. A crucial first step is to automatically standardize all column names (e.g., converting `Total Carbon wt%` to `total_carbon_wt_pct`) to ensure consistency.

2.  **Feature Selection**: The script intelligently separates the data into:
    * **Input Features**: The "raw ingredient" and processing columns (e.g., `graphite_wt_pct`, `mgo_purity_pct`, `firing_temp_c`).
    * **Target Labels**: The 8 material properties the model needs to learn to predict (e.g., `porosity_pct`, `density_g_cm3`).

3.  **Data Splitting**: The dataset is split into a training set (80%) and a test set (20%). The model **never sees the test set** during training. The test set is saved for final verification.

4.  **Model Training**: A **RandomForest Regressor** model is trained on the data. This is done within a scikit-learn `Pipeline`, which automatically handles data preprocessing steps like imputing missing values, scaling numeric features, and one-hot encoding categorical features.

5.  **Optimization**: `RandomizedSearchCV` is used to automatically tune the model's hyperparameters, ensuring the most accurate version of the model is selected.

6.  **Evaluation**: The final, optimized model is evaluated against the held-back test set. This provides an unbiased assessment of its real-world performance. All results from this process are saved in the `model_outputs` folder.

---

## 📊 Understanding the Output Folder (`model_outputs`)

This folder contains all the artifacts from the model training process.

* **`best_model.joblib`**: 🧠 This is the most important file. It's the final, trained model, ready to be used for making new predictions.
* **`test_set_for_verification.csv`**: A copy of the 20% of the data that the model was never trained on. This is used for final, unbiased evaluation.
* **`test_metrics_per_target.csv`**: The model's final "report card." It shows the performance scores (like R-squared) for each of the 8 properties.
* **`permutation_importances.csv` / `.png`**: These files show which input features (like firing temperature or graphite content) were most influential in the model's predictions. This is key for scientific insight.
* **`learning_curves/` & `residual_plots/`**: These folders contain diagnostic plots that help verify the model is learning correctly and that its errors are random (which is a good thing).

---

## 🚀 How to Use the Model for Predictions

You can use the trained model to predict the properties of new materials using the **`run_prediction.py`** script.

### 1. Requirements

Make sure you have the necessary Python libraries installed. You can install them using pip:
```sh
pip install pandas scikit-learn joblib matplotlib seaborn

2. Running the Script

Open a terminal or command prompt in the project folder and run the following command:
Bash

python run_prediction.py

3. Providing Input

The script will ask you to choose an input method:

    Manual Entry (Choice 1): The script will prompt you to enter the value for each of the required "raw ingredient" features one by one.

    File Input (Choice 2): The script will ask for the path to a CSV or Excel file. This file must contain the same "raw ingredient" columns as the training data.

4. Getting the Output

The script will display a table of the predicted properties directly on your screen. It will also save this table as a new file named prediction_results.csv in the main project folder for your records.