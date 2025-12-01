import pandas as pd
import numpy as np

print("🔍 DATASET INSPECTION")
print("="*50)

# Load and inspect the dataset
df = pd.read_csv('./EllipticPlusPlus-main/Actors Dataset/wallets_features_classes_combined.csv')

print(f"[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Dataset columns: {len(df.columns)}")

# Check class distribution
print(f"\n📊 CLASS DISTRIBUTION:")
print(df['class'].value_counts().sort_index())
print(f"Unique classes: {df['class'].unique()}")

# Check for missing values
print(f"\n❓ MISSING VALUES:")
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

if len(missing_summary) > 0:
    print("Columns with missing values:")
    for col, count in missing_summary.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"  {col}: {count:,} ({percentage:.1f}%)")
else:
    print("✅ No missing values found!")

# Check data types
print(f"\n📋 DATA TYPES:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# Sample a few rows
print(f"\n📄 SAMPLE DATA (first 3 rows):")
print(df.head(3))

# Check for any weird values in class column
print(f"\n🔍 CLASS COLUMN ANALYSIS:")
print(f"Class column type: {df['class'].dtype}")
print(f"Class unique values: {sorted(df['class'].unique())}")
print(f"Class value counts:")
for class_val in sorted(df['class'].unique()):
    count = sum(df['class'] == class_val)
    percentage = (count / len(df)) * 100
    print(f"  Class {class_val}: {count:,} ({percentage:.1f}%)")

# Check if there are any NaN values in class column
class_nan_count = df['class'].isnull().sum()
print(f"NaN values in class column: {class_nan_count}")

# Check for any non-numeric values in class column
print(f"\n🧮 NUMERIC CHECK:")
try:
    numeric_classes = pd.to_numeric(df['class'], errors='coerce')
    nan_after_conversion = numeric_classes.isnull().sum()
    print(f"Values that couldn't be converted to numeric: {nan_after_conversion}")
    if nan_after_conversion > 0:
        non_numeric_values = df[numeric_classes.isnull()]['class'].unique()
        print(f"Non-numeric values found: {non_numeric_values}")
except Exception as e:
    print(f"Error checking numeric conversion: {e}")

# Check what happens after filtering
print(f"\n🔬 FILTERING ANALYSIS:")
print("Step by step filtering...")

# Step 1: Original data
print(f"1. Original data: {len(df)} rows")
print(f"   Class distribution: {df['class'].value_counts().to_dict()}")

# Step 2: Remove NaN values
df_step2 = df.dropna()
print(f"2. After dropna(): {len(df_step2)} rows")
if len(df_step2) > 0:
    print(f"   Class distribution: {df_step2['class'].value_counts().to_dict()}")
else:
    print("   ⚠️  All rows removed by dropna()!")

# Step 3: Filter classes
if len(df_step2) > 0:
    df_step3 = df_step2[df_step2['class'].isin([0, 1])]
    print(f"3. After class filtering: {len(df_step3)} rows")
    if len(df_step3) > 0:
        print(f"   Class distribution: {df_step3['class'].value_counts().to_dict()}")
    else:
        print("   ⚠️  All rows removed by class filtering!")
        print("   Classes found in data before filtering:")
        print(f"   {df_step2['class'].unique()}")

print(f"\n💡 RECOMMENDATIONS:")
if class_nan_count > 0:
    print("- Handle NaN values in class column")
if len(missing_summary) > 0:
    print("- Consider filling missing values instead of dropping")
if len(df_step2) == 0:
    print("- dropna() is removing all data - use selective filling")
if len(df_step3) == 0:
    print("- Class filtering is wrong - check actual class values")