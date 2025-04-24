# Critical Temperature Prediction of Superconductors

## Project Description
This project develops machine learning models to predict the critical temperature of superconductors based on their chemical composition and physical properties. Superconductivity - the phenomenon where materials exhibit zero electrical resistance below a critical temperature - was discovered by Heike Kamerlingh Onnes in 1911. Despite being known for over a century, the relationship between superconductivity and material properties remains poorly understood.

**Competition:** [Critical Temperature of Superconductors](https://www.kaggle.com/c/critical-temperature-of-superconductors/overview) 

The project was developed as part of the Samsung Innovation Campus Bootcamp: Classical Machine Learning program, aiming to establish connections between chemical composition, various material properties, and critical temperature.

## Key Results
**Competition Score:** 88.71050  
**Best Model:** Random Forest Regressor  
**Top Metrics:**
- Mean Absolute Percentage Error: 3.31
- Mean Absolute Error: 5.49 K
- Root Mean Squared Error: 9.77 K

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
- Random Forest (MAPE: 3.31) ← **Selected as Final Model**
- SVR (MAPE: 6.66)
- KNN (MAPE: 2.92)
- Linear Regression (MAPE: 8.07)
- CatBoost (MAPE: 4.96)

### Final Model Architecture
```python
Pipeline([
    ('NewGenerator', NewGenerator()),  # Custom feature processor
    ('scaler', StandardScaler()),      # Feature normalization
    ('forest_model', RandomForestRegressor())  # Prediction model
])
