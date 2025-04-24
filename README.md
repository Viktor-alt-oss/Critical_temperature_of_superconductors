# Critical Temperature Prediction of Superconductors

## Project Description
This project develops machine learning models to predict the critical temperature of superconductors based on their chemical composition and physical properties. Superconductivity - the phenomenon where materials exhibit zero electrical resistance below a critical temperature - was discovered by Heike Kamerlingh Onnes in 1911. Despite being known for over a century, the relationship between superconductivity and material properties remains poorly understood.

The project was developed as part of the Samsung Innovation Campus Bootcamp: Classical Machine Learning program, aiming to establish connections between chemical composition, various material properties, and critical temperature.

## Dataset Description
The data comes from the superconducting materials database collected by Japan's National Institute for Materials Science (NIMS), containing information on 21,263 superconductors:

- **Training set**: 17,010 records
- **Test set**: 4,253 records

For each superconductor, the data includes:
- Full chemical formula
- 8 key chemical properties (absolute values, means, weighted means, etc.):
  - Atomic mass
  - Ionization energy
  - Atomic radius
  - Density
  - Fusion heat
  - Electron affinity
  - Thermal conductivity
  - Valence

### Files:
- `train.csv`: Training data with superconductor properties
- `formula_train.csv`: Training data with chemical composition
- `test.csv`: Test data with superconductor properties
- `formula_test.csv`: Test data with chemical composition

## Methodology
### Data Preprocessing
- Merged property and formula datasets
- Standardized features using StandardScaler
- Applied PCA for dimensionality reduction (14 components)

### Model Evaluation
Tested multiple regression approaches:
- Decision Tree (MAPE: 3.46)
- Random Forest (MAPE: 3.31)
- SVR (MAPE: 6.66)
- KNN (MAPE: 2.92)
- Linear Regression (MAPE: 8.07)
- CatBoost (MAPE: 4.96)

### Final Model
Selected Random Forest Regressor pipeline:
1. Custom feature generator
2. Standard scaling
3. Random Forest regression

## Results
Best performing model achieved:
- Mean Absolute Percentage Error: 3.31
- Mean Absolute Error: 5.49 K
- Root Mean Squared Error: 9.77 K

## How to Use
### Requirements
- Python 3.7+
- Libraries: pandas, numpy, scikit-learn, catboost, joblib

### Using the Trained Model
1. Load the model:
```python
from joblib import load
model = load('Critical_temperature_of_superconductors.joblib')
