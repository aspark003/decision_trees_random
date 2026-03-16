```markdown

# Decision Trees and Ensemble Regression Models

This project explores regression models for predicting housing prices using tree-based machine learning 
algorithms implemented in Python with scikit-learn.

The project compares the performance of three models:

- DecisionTreeRegressor

- ExtraTreesRegressor

- RandomForestRegressor

The goal is to evaluate how ensemble methods improve prediction accuracy compared to a single decision tree.

# Project Overview

The project builds a complete machine learning workflow that includes:

data preprocessing

feature transformation

model training

hyperparameter tuning

model evaluation

feature importance analysis

residual diagnostics

The dataset contains demographic and housing data used to predict the target variable median_house_value.


# Dataset

The dataset includes housing and demographic statistics such as:

Feature	Description
- longitude	
- latitude	
- housing_median_age	
- total_rooms	
- population
- households	
- median_income
- median_house_value	

Some columns were removed during preprocessing:

- total_bedrooms
- ocean_proximity

# Data Preprocessing

The preprocessing pipeline performs:

- Missing value handling

- Numeric features

```python
SimpleImputer(strategy="median")

# Categorical features

SimpleImputer(strategy="constant", fill_value="missing")

- Feature Scaling
- StandardScaler
- Categorical Encoding
- OneHotEncoder(drop="first")

```
All preprocessing steps are implemented using:

ColumnTransformer
Pipeline

This ensures the transformations are applied consistently during training and testing.

# Machine Learning Workflow

The model training process follows these steps:

- Load processed dataset

- Separate features and target variable

- Apply preprocessing pipeline

- Split dataset into training and testing sets

- Train regression model using GridSearchCV

- Generate predictions

- Evaluate model performance

- Analyze residuals

- Visualize feature importance

Model Evaluation Metrics

The models were evaluated using:

# Metric	Meaning

- R² Score	proportion of variance explained
- MAE	mean absolute error
- MSE	mean squared error
- RMSE	root mean squared error

# Model Results

# Decision Tree Regressor
R²: 0.7419
MAE: 38252
MSE: 3382460937
RMSE: 58158

The decision tree provides a baseline model but shows higher prediction error compared to ensemble methods.

# Extra Trees Regressor
R²: 0.7925
MAE: 35212
MSE: 2719487515
RMSE: 52148

Extra Trees improves performance by using an ensemble of randomized decision trees.

# Random Forest Regressor
R²: 0.8115
MAE: 32426
MSE: 2470173390
RMSE: 49700

Random Forest achieved the best performance among the evaluated models.

# Feature Importance

Across all models, the most influential features were:

# Feature	Importance
- median_income	strongest predictor
- longitude	geographic influence
- latitude	geographic influence
- population	local density
- total_rooms	housing characteristics

Median income consistently had the highest importance score, indicating a strong relationship with housing prices.

# Visualizations

The project generates the following plots:

Actual vs Predicted values

Predicted vs Residuals

Feature importance charts

These visualizations help analyze prediction accuracy and model behavior.

Project Structure
extra_tree
│
├── loader
│   ├── housing.csv
│   ├── extra_tree_process.pkl
│   └── main.py
│
├── processor
│   └── processor.py
│
├── model
│   └── decision_extra_random.py
│
├── plots
│
├── README.md
└── requirements.txt

# Installation

Clone the repository:

git clone https://github.com/aspark003/decision_trees_random.git

Create a virtual environment:

- python -m venv .venv

Activate it:

Windows

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Running the Project

Execute the model comparison script:

python model/decision_extra_random.py

The script will:

- train all models

- print evaluation metrics

- generate diagnostic plots

# Conclusion

This project demonstrates how ensemble tree methods improve predictive performance over a single decision tree model.

Among the models evaluated:

Decision Tree provides a baseline model

Extra Trees improves prediction stability

Random Forest achieves the highest predictive accuracy

The results highlight the effectiveness of ensemble learning for structured tabular data.



```