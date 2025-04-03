import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths to your data files
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')
sample_submission_path = os.path.join(data_dir, 'sample_submission.csv')

# Load the datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)

# Display first few rows of train data
print("\nTrain data head:")
print(train_df.head())

# Summary statistics for numerical columns
print("\nTrain data summary statistics:")
print(train_df.describe())

# Check missing values
print("\nMissing values in train data:")
print(train_df.isnull().sum())

# Check unique values in categorical columns
print("\nUnique kingdoms:", train_df['kingdom'].unique())
print("Number of unique kingdoms:", train_df['kingdom'].nunique())

# Check if temperature needs conversion from K to °C
if train_df['Avg_Temperature'].max() > 100:
    print("Converting temperatures from K to °C for analysis")
    # Make a copy to avoid modifying the original data
    analysis_df = train_df.copy()
    kelvin_columns = ['Avg_Temperature', 'Avg_Feels_Like_Temperature']
    for col in kelvin_columns:
        if col in analysis_df.columns:
            analysis_df[col] = analysis_df[col] - 273.15
else:
    analysis_df = train_df

# Display distribution of target variables
target_vars = ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']

# Create folders for plots if they don't exist
plots_dir = os.path.join(data_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Plot histograms of target variables
plt.figure(figsize=(15, 12))
for i, var in enumerate(target_vars):
    plt.subplot(3, 2, i+1)
    sns.histplot(analysis_df[var], kde=True)
    plt.title(f'Distribution of {var}')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'target_distributions.png'))
print("Saved target distributions plot")

# Plot monthly patterns of target variables
plt.figure(figsize=(15, 15))
monthly_avg = analysis_df.groupby('Month')[target_vars].mean()
for i, var in enumerate(target_vars):
    plt.subplot(3, 2, i+1)
    plt.plot(monthly_avg.index, monthly_avg[var])
    plt.title(f'Monthly Average {var}')
    plt.xlabel('Month')
    plt.ylabel(var)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'monthly_patterns.png'))
print("Saved monthly patterns plot")

# Check the correlation between variables
plt.figure(figsize=(12, 10))
correlation = analysis_df[target_vars + ['Temperature_Range', 'Rain_Duration', 'Evapotranspiration']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
print("Saved correlation matrix plot")

# Analyze data by kingdom
kingdom_stats = analysis_df.groupby('kingdom')[target_vars].mean()
print("\nAverage values by kingdom:")
print(kingdom_stats)

# Plot kingdom-based patterns
plt.figure(figsize=(15, 10))
for i, var in enumerate(target_vars):
    plt.subplot(3, 2, i+1)
    sns.boxplot(x='kingdom', y=var, data=analysis_df.sample(5000))  # Sample to reduce plot size
    plt.title(f'{var} by Kingdom')
    plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'kingdom_patterns.png'))
print("Saved kingdom patterns plot")

# Analyze geographical patterns
plt.figure(figsize=(15, 12))
for i, var in enumerate(target_vars):
    plt.subplot(3, 2, i+1)
    plt.scatter(analysis_df['longitude'], analysis_df['latitude'], 
               c=analysis_df[var], cmap='viridis', alpha=0.5, s=5)
    plt.colorbar(label=var)
    plt.title(f'Geographical Distribution of {var}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'geographical_patterns.png'))
print("Saved geographical patterns plot")

# Look for seasonal patterns
analysis_df['season'] = (analysis_df['Month'] % 12 + 3) // 3
season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
analysis_df['season_name'] = analysis_df['season'].map(season_names)

plt.figure(figsize=(15, 12))
for i, var in enumerate(target_vars):
    plt.subplot(3, 2, i+1)
    sns.boxplot(x='season_name', y=var, data=analysis_df)
    plt.title(f'{var} by Season')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'seasonal_patterns.png'))
print("Saved seasonal patterns plot")

# Save kingdom stats for later use
kingdom_stats.to_csv(os.path.join(data_dir, 'kingdom_stats.csv'))

print("Data exploration completed. Visualizations saved in the 'plots' folder.")

# Additional analysis for improving model performance
# Analyze Rain_Amount specifically as it was identified as challenging
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Radiation', y='Rain_Amount', hue='season_name', data=analysis_df.sample(5000))
plt.title('Radiation vs Rain Amount by Season')
plt.savefig(os.path.join(plots_dir, 'rain_analysis.png'))
print("Saved rain analysis plot")

# Analyze Wind patterns
plt.figure(figsize=(12, 8))
plt.scatter(analysis_df['Wind_Speed'] * np.cos(np.radians(analysis_df['Wind_Direction'])),
           analysis_df['Wind_Speed'] * np.sin(np.radians(analysis_df['Wind_Direction'])),
           alpha=0.1, s=1)
plt.title('Wind Vectors (Speed and Direction)')
plt.xlabel('East-West Component')
plt.ylabel('North-South Component')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'wind_vectors.png'))
print("Saved wind vectors plot")

# Analyze the distribution of values in the test set compared to train
# Get overlapping columns
common_cols = [col for col in test_df.columns if col in train_df.columns]
plt.figure(figsize=(15, 10))
for i, col in enumerate(common_cols):
    if col in ['ID', 'kingdom']:
        continue
    plt.subplot(3, 2, (i % 6) + 1)
    sns.histplot(train_df[col], alpha=0.5, label='Train', kde=True)
    sns.histplot(test_df[col], alpha=0.5, label='Test', kde=True)
    plt.title(f'Distribution of {col}')
    plt.legend()
    if (i+1) % 6 == 0 or i+1 == len(common_cols):
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'train_test_comparison_{i//6}.png'))
        if i+1 < len(common_cols):
            plt.figure(figsize=(15, 10))
print("Saved train-test comparison plots")

print("Enhanced data exploration completed.")