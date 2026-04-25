import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- PHASE 1: LOADING & INITIAL EXPLORATION ---
print("Loading 'Tips' dataset for EDA...")
df = sns.load_dataset('tips')

# 1. Statistical Summary
print("\n--- Statistical Summary ---")
print(df.describe())

# 2. Check Data Info
print("\n--- Data Structure ---")
print(df.info())

# --- PHASE 2: IDENTIFYING CORRELATIONS ---
# We need to convert categories to codes for the correlation matrix
print("\nCalculating Correlations...")
numeric_df = df.copy()
for col in ['sex', 'smoker', 'day', 'time']:
    numeric_df[col] = numeric_df[col].astype('category').cat.codes

plt.figure(figsize=(10, 8))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap: Identifying Influencing Factors')
plt.show()

# --- PHASE 3: PATTERN DISCOVERY ---

# Insight 1: Distribution of Total Bill by Day (Identifying Peak Trends)
plt.figure(figsize=(10, 6))
sns.boxenplot(data=df, x='day', y='total_bill', palette='Set2')
plt.title('Spending Patterns Across Different Days')
plt.show()

# Insight 2: Relationship between Total Bill and Tip (Influencing Factors)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='total_bill', y='tip', hue='time', style='time', s=100)
plt.title('Relationship: Bill Amount vs Tip (Lunch vs Dinner)')
plt.show()

# Insight 3: Pairwise relationships (The "Bird's Eye View")
# This creates a grid showing all relationships at once
print("\nGenerating Pair Plot (this may take a moment)...")
sns.pairplot(df, hue='sex', palette='husl')
plt.suptitle('Pairwise Relationship Overview by Gender', y=1.02)
plt.show()

# --- SUMMARY REPORT ---
print("\n--- EDA Insights Summary ---")
print("1. Correlation: There is a strong positive correlation between Total Bill and Tip.")
print("2. Trends: Spending (Total Bill) tends to be higher and more varied on weekends (Sat/Sun).")
print("3. Factors: Dinner time generally results in higher bills compared to lunch.")