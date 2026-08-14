# Student Performance Predictor 📊

A machine learning project that predicts a student's final marks based on academic and personal factors such as study hours, attendance, previous scores, assignment scores, gender, and parent education.

The project demonstrates an end-to-end machine learning workflow including data generation, data cleaning, exploratory data analysis (EDA), feature preprocessing, model training, prediction, and model evaluation.

## 🎯 Project Objective

Educational institutions can use predictive analysis to understand factors associated with student performance and identify students who may need additional academic support.

This project compares multiple regression algorithms to predict **Final Marks**.

## 📁 Project Structure

```text
student-performance-predictor/
│
├── code/
│   ├── 01_generate_data.py
│   ├── 02_eda_preprocessing.py
│   └── 03_model_training.py
│
├── data/
│   ├── student_performance.csv
│   └── student_performance_cleaned.csv
│
├── models/
│   ├── linear_regression.pkl
│   ├── decision_tree_regressor.pkl
│   ├── random_forest_regressor.pkl
│   ├── scaler.pkl
│   └── model_comparison.csv
│
└── plots/
    ├── 01_correlation_heatmap.png
    ├── 02_study_hours_vs_marks.png
    ├── 03_attendance_vs_marks.png
    ├── 04_final_marks_distribution.png
    ├── 05_marks_by_parent_education.png
    ├── 06_marks_by_gender.png
    ├── 07_feature_importance.png
    ├── 08_predicted_vs_actual.png
    └── 09_model_comparison.png
```

## 📊 Dataset

The project uses a **synthetically generated dataset containing 500 student records**.

### Features

| Feature             | Description                      |
| ------------------- | -------------------------------- |
| `Study_Hours`       | Number of hours spent studying   |
| `Attendance`        | Student attendance percentage    |
| `Previous_Score`    | Previous academic score          |
| `Assignments_Score` | Assignment performance           |
| `Gender`            | Student gender                   |
| `Parent_Education`  | Parent's highest education level |
| `Final_Marks`       | Target variable                  |

Missing values were intentionally introduced into selected numerical features to simulate real-world data-cleaning requirements.

> **Note:** The dataset is synthetic and was created specifically for this project. It should not be interpreted as real-world student data.

## 🔄 Machine Learning Workflow

The project follows these main steps:

### 1. Data Generation

A synthetic dataset is generated using NumPy and Pandas with realistic relationships between academic factors and final marks.

### 2. Data Preprocessing

The project:

* Handles missing numerical values using median imputation
* Encodes categorical variables
* Creates a cleaned dataset for modeling

### 3. Exploratory Data Analysis

Several visualizations are generated to investigate relationships between features and final marks, including:

* Correlation heatmap
* Study hours vs. final marks
* Attendance vs. final marks
* Final marks distribution
* Marks by parent education
* Marks by gender
* Random Forest feature importance
* Predicted vs. actual values
* Model comparison

### 4. Model Training

Three regression algorithms were trained and compared:

* **Linear Regression**
* **Decision Tree Regressor**
* **Random Forest Regressor**

The data was divided into training and testing sets using an 80/20 split.

### 5. Model Evaluation

The models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## 📈 Model Results

| Model                   |     MAE ↓ |    RMSE ↓ | R² Score ↑ |
| ----------------------- | --------: | --------: | ---------: |
| **Linear Regression**   | **4.919** | **5.950** |  **0.674** |
| Random Forest Regressor |     5.017 |     6.255 |      0.640 |
| Decision Tree Regressor |     6.101 |     7.855 |      0.432 |

### 🏆 Best Performing Model

**Linear Regression** achieved the highest R² score (**0.674**) and the lowest MAE (**4.919**) and RMSE (**5.950**) among the three tested models.

This demonstrates why comparing multiple models is useful instead of automatically selecting the most complex algorithm.

## 📌 Key Analysis

The exploratory analysis indicates that **Study Hours** has the strongest relationship with Final Marks among the included features.

The Random Forest model's feature-importance analysis was also used to understand the relative contribution of the input features.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yash-178/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Run the project

Run the scripts in order:

```bash
python code/01_generate_data.py
python code/02_eda_preprocessing.py
python code/03_model_training.py
```

The generated cleaned dataset, trained models, model comparison results, and visualizations will be stored in their respective directories.

## 🎓 Learning Outcomes

Through this project, the following machine learning concepts were practiced:

* Data preprocessing
* Missing-value handling
* Categorical encoding
* Exploratory data analysis
* Train-test splitting
* Feature scaling
* Regression
* Model comparison
* Model evaluation
* Feature importance
* Prediction analysis

## 👨‍💻 Author

**Yash Chaturvedi**

MCA Graduate | Aspiring Data Analyst / Data Scientist

---

⭐ If you found this project useful, feel free to explore the repository and its visualizations.
