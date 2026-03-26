# Agriculture Crop Production Prediction in India

## Project Overview
This project analyzes agricultural data from India (2001-2014) to predict crop yields based on cultivation costs, production costs, and crop types. The goal is to assist in understanding the factors driving agricultural productivity.

## Data Processing
The raw data was split across multiple files. We performed the following steps to create a unified dataset:

1.  **Data Sourcing**:
    *   `datafile (1).csv`: State-wise Cost of Cultivation and Yield.
    *   `datafile (2).csv`: National-level Production time-series (reshaped from wide to long format).
    *   `datafile (3).csv`: Crop Variety and Recommended Zones.
2.  **Cleaning & Merging**:
    *   Standardized crop names (e.g., "Paddy" $\rightarrow$ "rice", "Arhar" $\rightarrow$ "tur").
    *   Merged datasets on `Crop` and `State`.
    *   handled missing values in `Variety` and `Season` columns.
    *   **Output**: `merged_crop_data.csv` (220 rows, 12 columns).

## Exploratory Data Analysis (EDA)
Key findings from the data analysis:
*   **High Correlation**: Cultivation Cost (`cost_c2`) is strongly correlated ($>0.9$) with Yield. Higher investment generally leads to higher output.
*   **Yield Variation**: Sugarcane has significantly higher yields (tons/hectare) compared to grains, acting as a natural outlier that the model must handle.
*   **Cost Efficiency**: Lower Cost of Production per quintal (`cost_production_c2`) is associated with higher yields, indicating economies of scale.

## Model Building & Results
We trained machine learning models to predict **Yield (Quintal/Hectare)**.

### Models Trained
1.  **Linear Regression**: Baseline model.
2.  **Random Forest Regressor**: Ensemble model to capture non-linear relationships and crop-specific traits (like Sugarcane vs. Rice).

### Performance
| Model | R² Score | MAE (Mean Absolute Error) |
| :--- | :--- | :--- |
| Linear Regression | 0.9861 | 18.52 |
| **Random Forest** | **0.9990** | **2.09** |

The **Random Forest** model achieved near-perfect accuracy ($R^2=0.999$), identifying **Cost of Production** and **Crop Type** as the most critical predictors.

## Project Files
*   `Crop_Prediction_Analysis.ipynb`: **Jupyter Notebook** with step-by-step code and markdown explanations.
*   `crop_prediction_pipeline.py`: **Main script** containing the full workflow (Loading, Cleaning, Merging, EDA, Training, Evaluation).
*   `merged_crop_data.csv`: The cleaned and merged dataset used for modeling.
*   `prediction_plot.png`: Visualization of Actual vs. Predicted yields from the best model.
*   `model_report.md`: Detailed technical report on model parameters and feature importance.
*   `README.md`: Project documentation.

## How to Run
1.  Install dependencies: `pip install pandas scikit-learn matplotlib seaborn`
2.  Run the pipeline: `python crop_prediction_pipeline.py`
3.  Check the output `merged_crop_data.csv` and `prediction_plot.png`.
