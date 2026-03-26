import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True).str.lower()
    return df

def clean_crop_name(name):
    if pd.isna(name): return name
    name = str(name).lower().strip()
    # Normalize common crop names
    if 'paddy' in name: return 'rice'
    if 'arhar' in name: return 'tur' # Arhar is Tur
    if 'gram' in name: return 'gram'
    if 'groundnut' in name: return 'groundnut'
    if 'maize' in name: return 'maize'
    return name

print("--- Step 1: Loading and Cleaning Data ---")

# Load Data
df1 = pd.read_csv("datafile (1).csv") # State/Cost
df2 = pd.read_csv("datafile (2).csv") # Production Time Series
df3 = pd.read_csv("datafile (3).csv") # Varieties

# Clean DF2 (Production)
df2 = df2.rename(columns={df2.columns[0]: 'crop'})
id_vars = ['crop']
df2_long = pd.DataFrame()
metrics = ['Production', 'Area', 'Yield']

print("Reshaping time-series data...")
for metric in metrics:
    cols = [c for c in df2.columns if metric in c and 'crop' not in c]
    temp = df2.melt(id_vars=['crop'], value_vars=cols, var_name='Year_Raw', value_name=metric)
    temp['Year'] = temp['Year_Raw'].apply(lambda x: re.search(r'\d{4}-\d{2}', x).group(0) if re.search(r'\d{4}-\d{2}', x) else None)
    temp = temp.drop(columns=['Year_Raw'])
    
    if df2_long.empty:
        df2_long = temp
    else:
        df2_long = pd.merge(df2_long, temp, on=['crop', 'Year'], how='outer')

df2_long['crop'] = df2_long['crop'].apply(clean_crop_name)

# Clean DF1 (Cost)
df1 = clean_column_names(df1)
df1 = df1.rename(columns={
    'cost_of_cultivation_(`/hectare)_a2+fl': 'cost_a2_fl',
    'cost_of_cultivation_(`/hectare)_c2': 'cost_c2',
    'cost_of_production_(`/quintal)_c2': 'cost_production_c2',
    'yield_(quintal/_hectare)_': 'yield_state'
})
df1['crop'] = df1['crop'].apply(clean_crop_name)

# Clean DF3 (Varieties)
df3 = clean_column_names(df3)
rows = []
for idx, row in df3.iterrows():
    if pd.notna(row['recommended_zone']):
        # Clean zone string
        zone_str = re.sub(r'under\s+.*', '', row['recommended_zone'], flags=re.IGNORECASE)
        zone_str = zone_str.replace(' and ', ',')
        states = [s.strip() for s in zone_str.split(',') if s.strip()]
        for state in states:
            new_row = row.to_dict()
            new_row['state'] = state
            rows.append(new_row)
df3_exploded = pd.DataFrame(rows)
if not df3_exploded.empty:
    df3_exploded = df3_exploded.drop(columns=['recommended_zone', 'unnamed:_4'])
    df3_exploded['crop'] = df3_exploded['crop'].apply(clean_crop_name)
    df3_exploded['state'] = df3_exploded['state'].str.strip()

# Merge
print("Merging datasets...")
# Merge Time Series (National) with Cost (State)
merged_df = pd.merge(df2_long, df1, on='crop', how='inner')

# Merge with Varieties (Optional left join)
if not df3_exploded.empty:
    final_df = pd.merge(merged_df, df3_exploded, on=['crop', 'state'], how='left')
else:
    final_df = merged_df

print(f"Merged Data Shape: {final_df.shape}")
final_df.to_csv("merged_crop_data.csv", index=False)

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\n--- Step 2: Exploratory Data Analysis ---")
print(final_df.describe())

# Correlation Matrix
numeric_cols = final_df.select_dtypes(include=[np.number]).columns
corr = final_df[numeric_cols].corr()
print("\nCorrelation with Yield (State):")
if 'yield_(quintal/_hectare)' in corr.columns:
    print(corr['yield_(quintal/_hectare)'].sort_values(ascending=False))

# --- EDA Visualizations ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Correlation Heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[0, 0], cbar_kws={'shrink': 0.8})
axes[0, 0].set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# Plot 2: Yield Distribution by Crop
sns.boxplot(data=final_df, x='crop', y='yield_(quintal/_hectare)', palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('Yield Distribution by Crop Type', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Crop')
axes[0, 1].set_ylabel('Yield (Quintal/Hectare)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Cost vs Yield Scatter
sns.scatterplot(data=final_df, x='cost_c2', y='yield_(quintal/_hectare)', hue='crop', 
                palette='Set2', alpha=0.7, ax=axes[1, 0])
axes[1, 0].set_title('Cost of Cultivation vs Yield', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Cost of Cultivation (₹/Hectare)')
axes[1, 0].set_ylabel('Yield (Quintal/Hectare)')
axes[1, 0].legend(title='Crop', bbox_to_anchor=(1.02, 1), loc='upper left')

# Plot 4: Average Yield by State
state_yield = final_df.groupby('state')['yield_(quintal/_hectare)'].mean().sort_values(ascending=True)
state_yield.plot(kind='barh', color='steelblue', ax=axes[1, 1])
axes[1, 1].set_title('Average Yield by State', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Average Yield (Quintal/Hectare)')
axes[1, 1].set_ylabel('State')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=150, bbox_inches='tight')
print("EDA plots saved to eda_visualizations.png")

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print("\n--- Step 3: Model Training ---")

# Features & Target
# We use Cost and Year to predict State Yield
feature_cols = ['cost_a2_fl', 'cost_c2', 'cost_production_c2', 'Year', 'state', 'crop']
target_col = 'yield_(quintal/_hectare)'

X = final_df[feature_cols].copy()
y = final_df[target_col]

# Preprocessing
X['Year'] = X['Year'].astype(str).str.split('-').str[0].astype(int)

# Pipeline
categorical_features = ['state', 'crop']
numerical_features = ['cost_a2_fl', 'cost_c2', 'cost_production_c2', 'Year']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

print("Training Random Forest Regressor...")
pipeline.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
print("\n--- Step 4: Evaluation ---")
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# --- Model Evaluation Visualizations ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.7, c='steelblue', edgecolors='black', linewidth=0.5)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel("Actual Yield (Quintal/Hectare)", fontsize=10)
axes[0].set_ylabel("Predicted Yield (Quintal/Hectare)", fontsize=10)
axes[0].set_title(f"Actual vs Predicted Yield\nR² = {r2:.4f}", fontsize=12, fontweight='bold')
axes[0].legend()

# Plot 2: Residuals (Error Distribution)
residuals = y_test - y_pred
axes[1].hist(residuals, bins=20, color='coral', edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--', lw=2)
axes[1].set_xlabel("Prediction Error (Actual - Predicted)", fontsize=10)
axes[1].set_ylabel("Frequency", fontsize=10)
axes[1].set_title(f"Residuals Distribution\nMAE = {mae:.2f}", fontsize=12, fontweight='bold')

# Plot 3: Feature Importance
rf_model = pipeline.named_steps['model']
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(cat_feature_names)
importances = rf_model.feature_importances_

# Get top 10 features
indices = np.argsort(importances)[::-1][:10]
top_features = [all_feature_names[i] for i in indices]
top_importances = importances[indices]

axes[2].barh(range(len(top_features)), top_importances[::-1], color='seagreen', edgecolor='black')
axes[2].set_yticks(range(len(top_features)))
axes[2].set_yticklabels(top_features[::-1])
axes[2].set_xlabel("Feature Importance", fontsize=10)
axes[2].set_title("Top 10 Important Features", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("prediction_plot.png", dpi=150, bbox_inches='tight')
print("Prediction plots saved to prediction_plot.png")
