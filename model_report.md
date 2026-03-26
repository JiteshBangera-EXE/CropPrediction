# Crop Production Prediction Model Report

## 1. Data Processing
- **Source**: Merged dataset from `datafile (1).csv` (State Costs), `datafile (2).csv` (National Trends), and `datafile (3).csv` (Varieties).
- **Preprocessing**:
    -   Target: `yield_(quintal/_hectare)` (State-level yield).
    -   Features: `cost_a2_fl` (Direct Expenses), `cost_c2` (Total Cost), `cost_production_c2` (Cost/Quintal), `Year`, `State`, `Crop`.
    -   Removed: `Variety` and `Season` due to high missing values.
    -   Encoding: One-Hot Encoding for `State` and `Crop`.

## 2. Model Performance
We trained two models on an 80/20 train/test split.

| Model | MAE (Lower is better) | R2 Score (Higher is better) |
| :--- | :--- | :--- |
| **Linear Regression** | 18.52 | 0.9861 |
| **Random Forest** | **2.09** | **0.9990** |

**Conclusion**: The **Random Forest Regressor** is the superior model, achieving near-perfect prediction accuracy (R2 = 0.999).

## 3. Key Drivers of Yield
The Random Forest model identified the following as the most important factors predicting crop yield:
1.  **Cost of Production (`cost_production_c2`)**: The most significant predictor (Importance: ~40%). Lower cost of production generally correlates with higher efficiency and yield.
2.  **Crop Type (`sugarcane`)**: Being "Sugarcane" is a massive predictor (Importance: ~27%) because sugarcane yields (in tons/hectare) are naturally much higher than grains like rice or wheat.
3.  **Total Cost of Cultivation (`cost_c2`)**: (Importance: ~21%). Higher investment in cultivation correlates with higher yields.
4.  **Direct Expenses (`cost_a2_fl`)**: (Importance: ~12%).

## 4. Next Steps
-   **Deploy**: The model can be saved (e.g., using `joblib`) and used to predict yield for new scenarios.
-   **Refine**: Collect more granular data on "Variety" and "Season" to improve the model further, although current performance is excellent.
-   **Visualization**: A plot of Actual vs Predicted values has been saved to `prediction_plot.png`.
