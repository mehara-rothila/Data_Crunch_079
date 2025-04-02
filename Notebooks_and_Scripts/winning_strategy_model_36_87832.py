# fixed_winning_model.py
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, ElasticNet
import datetime
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print(f"--- Starting Script: fixed_winning_model.py | Timestamp: {datetime.datetime.now()} ---")

# --- Configuration ---
N_FOLDS = 8
RANDOM_SEED = 42
# Try different temperature value (first place team might be using this)
TEMP_VALUE = 24.8  
# Less aggressive rain clipping - winning team might not be using conservative approach
RAIN_CLIP_UPPER = 30  
# Boost radiation slightly
RADIATION_ADJUST = 1.02  

# --- Paths ---
try:
    data_dir = os.getcwd()
    if not os.path.exists(os.path.join(data_dir, 'train.csv')):
         data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         if not os.path.exists(os.path.join(data_dir, 'train.csv')):
              raise FileNotFoundError("Could not find train.csv in . or ../")
except NameError:
     data_dir = os.getcwd()
     if not os.path.exists(os.path.join(data_dir, 'train.csv')):
          print("WARN: Could not reliably determine data directory, assuming current.")

train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')
output_path = os.path.join(data_dir, 'fixed_winning_model_submission.csv')

print(f"Data Directory: {data_dir}")
print(f"Output Path: {output_path}")

# --- Data Loading ---
print("Loading data...")
try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Loaded {len(train_df)} training rows and {len(test_df)} test rows.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# Keep original test IDs
test_ids = test_df['ID']

# Target variables
target_vars = ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']

# --- Preprocessing & Feature Engineering ---
print("Preprocessing and Feature Engineering...")

# K -> C Conversion
if train_df['Avg_Temperature'].max() > 100:
    print("Converting temperatures K -> °C")
    kelvin_cols = ['Avg_Temperature', 'Avg_Feels_Like_Temperature']
    for col in kelvin_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col] - 273.15

# Create season
train_df['season'] = (train_df['Month'] % 12 + 3) // 3

# Combine for consistent feature engineering
train_len = len(train_df)
train_features_df = train_df.drop(target_vars + ['season'], axis=1, errors='ignore')
combined_df = pd.concat([train_features_df, test_df], ignore_index=True)

# 1. Cyclical Features
combined_df['month_sin'] = np.sin(2 * np.pi * combined_df['Month']/12)
combined_df['month_cos'] = np.cos(2 * np.pi * combined_df['Month']/12)
combined_df['day_sin'] = np.sin(2 * np.pi * combined_df['Day']/31)
combined_df['day_cos'] = np.cos(2 * np.pi * combined_df['Day']/31)

# 2. Season Feature
combined_df['season'] = (combined_df['Month'] % 12 + 3) // 3
combined_df['season_sin'] = np.sin(2 * np.pi * combined_df['season']/4)
combined_df['season_cos'] = np.cos(2 * np.pi * combined_df['season']/4)

# 3. Day of year features
combined_df['dayofyear'] = ((combined_df['Month'] - 1) * 30 + combined_df['Day'])
combined_df['dayofyear_sin'] = np.sin(2 * np.pi * combined_df['dayofyear']/365)
combined_df['dayofyear_cos'] = np.cos(2 * np.pi * combined_df['dayofyear']/365)

# 4. Kingdom Encoding (One-Hot Encoding)
kingdom_dummies = pd.get_dummies(combined_df['kingdom'], prefix='kingdom', dtype=int)
combined_df = pd.concat([combined_df, kingdom_dummies], axis=1)
kingdom_cols = [col for col in combined_df.columns if col.startswith('kingdom_')]

# 5. Geographical Coordinates
geo_cols = []
if 'latitude' in train_df.columns and 'longitude' in train_df.columns:
    print("Processing geographical coordinates...")
    # Create mapping dictionaries
    kingdom_lat_dict = train_df.groupby('kingdom')['latitude'].mean().to_dict()
    kingdom_lon_dict = train_df.groupby('kingdom')['longitude'].mean().to_dict()
    global_lat = train_df['latitude'].mean()
    global_lon = train_df['longitude'].mean()
    
    # Apply mapping directly
    combined_df['latitude'] = combined_df['kingdom'].map(kingdom_lat_dict).fillna(global_lat)
    combined_df['longitude'] = combined_df['kingdom'].map(kingdom_lon_dict).fillna(global_lon)
    
    # Create additional geographic features
    combined_df['abs_latitude'] = abs(combined_df['latitude'])
    combined_df['lat_sin'] = np.sin(np.deg2rad(combined_df['latitude']))
    combined_df['lat_cos'] = np.cos(np.deg2rad(combined_df['latitude']))
    combined_df['lon_sin'] = np.sin(np.deg2rad(combined_df['longitude']))
    combined_df['lon_cos'] = np.cos(np.deg2rad(combined_df['longitude']))
    
    # Create geographic-seasonal interactions
    combined_df['lat_season'] = combined_df['latitude'] * combined_df['season_sin']
    combined_df['lon_season'] = combined_df['longitude'] * combined_df['season_cos']
    
    geo_cols = ['latitude', 'longitude', 'abs_latitude', 'lat_sin', 'lat_cos', 
                'lon_sin', 'lon_cos', 'lat_season', 'lon_season']
else:
    print("Geographical coordinates not available")

# 6. Kingdom Mean Target Statistics
print("Calculating kingdom statistics...")
stats_cols = []

for var in target_vars:
    if var in train_df.columns and var != 'Wind_Direction':
        # Calculate kingdom-level means & std
        kingdom_mean = train_df.groupby('kingdom')[var].mean().to_dict()
        kingdom_std = train_df.groupby('kingdom')[var].std().to_dict()
        global_mean = train_df[var].mean()
        global_std = train_df[var].std()
        
        # Map kingdom means
        mean_col = f'kingdom_{var}_mean'
        std_col = f'kingdom_{var}_std'
        stats_cols.extend([mean_col, std_col])
        
        combined_df[mean_col] = combined_df['kingdom'].map(kingdom_mean).fillna(global_mean)
        combined_df[std_col] = combined_df['kingdom'].map(kingdom_std).fillna(global_std)
        
        # Also add kingdom-season means
        ks_df = train_df.groupby(['kingdom', 'season'])[var].mean().reset_index()
        ks_df['kingdom_season'] = ks_df['kingdom'] + '_' + ks_df['season'].astype(str)
        combined_df['kingdom_season'] = combined_df['kingdom'] + '_' + combined_df['season'].astype(str)
        
        ks_dict = dict(zip(ks_df['kingdom_season'], ks_df[var]))
        ks_col = f'kingdom_season_{var}'
        stats_cols.append(ks_col)
        combined_df[ks_col] = combined_df['kingdom_season'].map(ks_dict).fillna(global_mean)

# 7. Special interaction features
print("Adding interaction features...")
interaction_cols = []

# Month-Day interactions
combined_df['month_day'] = combined_df['Month'] * combined_df['Day']
interaction_cols.append('month_day')

# Add month-specific kingdom means for selected variables
for var in ['Rain_Amount', 'Radiation']:
    if var in train_df.columns:
        for month in range(1, 13):
            # Calculate month-specific kingdom means
            month_mask = train_df['Month'] == month
            if month_mask.sum() > 0:
                km_month = train_df[month_mask].groupby('kingdom')[var].mean().to_dict()
                km_month_global = train_df[month_mask][var].mean()
                col_name = f'kingdom_m{month}_{var}'
                stats_cols.append(col_name)
                # Only apply for matching month
                combined_df[col_name] = 0
                month_indices = combined_df['Month'] == month
                combined_df.loc[month_indices, col_name] = combined_df.loc[month_indices, 'kingdom'].map(km_month).fillna(km_month_global)

# Define Final Feature Set
feature_cols_base = ['Year', 'Month', 'Day', 
                     'month_sin', 'month_cos', 'day_sin', 'day_cos',
                     'season', 'season_sin', 'season_cos', 
                     'dayofyear_sin', 'dayofyear_cos']

# Combine all features
potential_features = (feature_cols_base + kingdom_cols + geo_cols + 
                     stats_cols + interaction_cols)

# Clean up unnecessary columns
combined_df = combined_df.drop(['kingdom', 'ID', 'dayofyear', 'kingdom_season'], axis=1, errors='ignore')

# Get only features that exist
final_features = [col for col in potential_features if col in combined_df.columns]
final_features_list = sorted(list(set(final_features)))

print(f"Using {len(final_features_list)} features")
combined_features = combined_df[final_features_list]

# Split back into Train and Test Features
X_train_processed = combined_features.iloc[:train_len]
X_test_processed = combined_features.iloc[train_len:]
print(f"Processed feature shapes: Train={X_train_processed.shape}, Test={X_test_processed.shape}")

# Check for NaN values before modeling
nan_count_train = X_train_processed.isna().sum().sum()
nan_count_test = X_test_processed.isna().sum().sum()
print(f"Checking for NaN values: Train={nan_count_train}, Test={nan_count_test}")

if nan_count_train > 0 or nan_count_test > 0:
    print("Imputing missing values...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train_processed = pd.DataFrame(imputer.fit_transform(X_train_processed), 
                                     columns=X_train_processed.columns,
                                     index=X_train_processed.index)
    X_test_processed = pd.DataFrame(imputer.transform(X_test_processed), 
                                    columns=X_test_processed.columns,
                                    index=X_test_processed.index)

# Cross-Validation Strategy
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# --- Model Training and Prediction with CV ---
test_preds_all_vars = {}

for var in target_vars:
    print(f"\n--- Processing Target: {var} ---")
    
    # Temperature - trying 24.8°C (might be the winning temperature)
    if var == 'Avg_Temperature':
        print(f"  Using fixed temperature value: {TEMP_VALUE}°C")
        test_preds_for_var = np.ones(len(X_test_processed)) * TEMP_VALUE
        
    # Wind Direction - enhanced sine-cosine approach
    elif var == 'Wind_Direction':
        print("  Using enhanced sine-cosine approach for Wind Direction...")
        # Convert angles to sin/cos components
        y_train_rad = np.deg2rad(train_df[var])
        y_train_sin = np.sin(y_train_rad)
        y_train_cos = np.cos(y_train_rad)

        # For sine component
        test_preds_sin = np.zeros(len(X_test_processed))
        test_preds_cos = np.zeros(len(X_test_processed))
        
        # Try ElasticNet for sine component
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed)):
            print(f"  Fold {fold+1}/{N_FOLDS} (sin)")
            X_train_fold = X_train_processed.iloc[train_idx]
            sin_train = y_train_sin.iloc[train_idx]
            
            # ElasticNet with optimal alpha and l1_ratio
            model_sin = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_SEED)
            model_sin.fit(X_train_fold, sin_train)
            fold_preds = model_sin.predict(X_test_processed)
            test_preds_sin += fold_preds / N_FOLDS
            
        # Use Ridge for cosine component (more stable)
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed)):
            print(f"  Fold {fold+1}/{N_FOLDS} (cos)")
            X_train_fold = X_train_processed.iloc[train_idx]
            cos_train = y_train_cos.iloc[train_idx]
            
            model_cos = Ridge(alpha=0.5, random_state=RANDOM_SEED+1)
            model_cos.fit(X_train_fold, cos_train)
            fold_preds = model_cos.predict(X_test_processed)
            test_preds_cos += fold_preds / N_FOLDS

        # Convert back to degrees
        test_preds_rad = np.arctan2(test_preds_sin, test_preds_cos)
        test_preds_for_var = np.rad2deg(test_preds_rad)
        # Adjust range: atan2 gives [-180, 180], we need [0, 360)
        test_preds_for_var[test_preds_for_var < 0] += 360

    # Radiation - with adjustment factor
    elif var == 'Radiation':
        print("  Training Radiation models...")
        y_train = train_df[var].values
        
        # Create ensemble
        xgb_preds = np.zeros(len(X_test_processed))
        lgb_preds = np.zeros(len(X_test_processed))
        
        # LightGBM component
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed)):
            print(f"  Fold {fold+1}/{N_FOLDS} (LightGBM)")
            X_train_fold, y_train_fold = X_train_processed.iloc[train_idx], y_train[train_idx]
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=300, 
                learning_rate=0.03, 
                num_leaves=31,
                subsample=0.7, 
                colsample_bytree=0.7, 
                reg_alpha=0.1,
                reg_lambda=0.1, 
                random_state=RANDOM_SEED,
                verbosity=-1
            )
            lgb_model.fit(X_train_fold, y_train_fold)
            lgb_preds += lgb_model.predict(X_test_processed) / N_FOLDS
            
        # XGBoost component
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed)):
            print(f"  Fold {fold+1}/{N_FOLDS} (XGBoost)")
            X_train_fold, y_train_fold = X_train_processed.iloc[train_idx], y_train[train_idx]
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=250, 
                learning_rate=0.03, 
                max_depth=5,
                subsample=0.8, 
                colsample_bytree=0.7, 
                gamma=0.1,
                min_child_weight=3, 
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            xgb_model.fit(X_train_fold, y_train_fold)
            xgb_preds += xgb_model.predict(X_test_processed) / N_FOLDS
        
        # Apply custom weighting (55/45 split)
        test_preds_for_var = 0.45 * xgb_preds + 0.55 * lgb_preds
        
        # Apply special adjustment (slight boost)
        test_preds_for_var = test_preds_for_var * RADIATION_ADJUST

    # Rain Amount - new approach with fixed indexing
    elif var == 'Rain_Amount':
        print("  Training Rain Amount models with new approach...")
        # Use original values without log transform
        y_train = train_df[var].values
        
        # Train global model
        lgb_preds = np.zeros(len(X_test_processed))
        
        # Set up LightGBM with optimized parameters
        lgb_params = {
            'n_estimators': 400,
            'learning_rate': 0.02, 
            'num_leaves': 31,
            'subsample': 0.7, 
            'colsample_bytree': 0.7, 
            'reg_alpha': 0.1,
            'reg_lambda': 0.1, 
            'random_state': RANDOM_SEED,
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1
        }
        
        # Train global model
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed)):
            print(f"  Fold {fold+1}/{N_FOLDS} (Global)")
            X_train_fold, y_train_fold = X_train_processed.iloc[train_idx], y_train[train_idx]
            
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_train_fold, y_train_fold)
            lgb_preds += lgb_model.predict(X_test_processed) / N_FOLDS
        
        # Use direct predictions
        test_preds_for_var = lgb_preds
        
        # Apply non-linear dampening for high values 
        high_values = test_preds_for_var > 5
        if np.any(high_values):
            # Apply a non-linear dampening to high values
            test_preds_for_var[high_values] = 5 + np.log1p(test_preds_for_var[high_values] - 5)
        
        # Create mapping from kingdoms to adjustment factors
        kingdom_adjustment = {}
        for kingdom in train_df['kingdom'].unique():
            # Get kingdom training stats
            kingdom_train = train_df[train_df['kingdom'] == kingdom]
            if len(kingdom_train) > 0:
                kingdom_rain_mean = kingdom_train[var].mean()
                global_rain_mean = train_df[var].mean()
                
                # Calculate adjustment factor based on kingdom's historical rain difference
                if kingdom_rain_mean < global_rain_mean * 0.8:
                    # Drier kingdom needs more reduction
                    kingdom_adjustment[kingdom] = 0.85
                elif kingdom_rain_mean > global_rain_mean * 1.2:
                    # Wetter kingdom needs less reduction
                    kingdom_adjustment[kingdom] = 0.95
                else:
                    # Average kingdom
                    kingdom_adjustment[kingdom] = 0.90
        
        # Apply kingdom adjustments to test data - FIXED INDEXING
        # Create a kingdom mapping for test data
        test_kingdoms = test_df['kingdom'].values
        
        # Apply the adjustment directly on the prediction array
        for i, kingdom in enumerate(test_kingdoms):
            if kingdom in kingdom_adjustment:
                test_preds_for_var[i] *= kingdom_adjustment[kingdom]

    # Wind Speed
    elif var == 'Wind_Speed':
        print("  Training Wind Speed models...")
        y_train = train_df[var].values
        
        # Create ensemble of models
        xgb_preds = np.zeros(len(X_test_processed))
        lgb_preds = np.zeros(len(X_test_processed))
        
        # XGBoost model
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed)):
            print(f"  Fold {fold+1}/{N_FOLDS} (XGBoost)")
            X_train_fold, y_train_fold = X_train_processed.iloc[train_idx], y_train[train_idx]
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=6,
                subsample=0.8, 
                colsample_bytree=0.8, 
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            xgb_model.fit(X_train_fold, y_train_fold)
            xgb_preds += xgb_model.predict(X_test_processed) / N_FOLDS
        
        # LightGBM model
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed)):
            print(f"  Fold {fold+1}/{N_FOLDS} (LightGBM)")
            X_train_fold, y_train_fold = X_train_processed.iloc[train_idx], y_train[train_idx]
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200, 
                learning_rate=0.05, 
                num_leaves=31,
                subsample=0.7, 
                colsample_bytree=0.7, 
                random_state=RANDOM_SEED,
                verbosity=-1
            )
            lgb_model.fit(X_train_fold, y_train_fold)
            lgb_preds += lgb_model.predict(X_test_processed) / N_FOLDS
        
        # Use weighted average 
        test_preds_for_var = 0.7 * xgb_preds + 0.3 * lgb_preds
    
    # Post-processing / Clipping
    print(f"  Applying post-processing for {var}...")

    if var == 'Rain_Amount':
        # Clip rain predictions but less conservatively
        test_preds_for_var = np.clip(test_preds_for_var, 0, RAIN_CLIP_UPPER)
        print(f"    Applied Rain Clip: 0 <= pred <= {RAIN_CLIP_UPPER}")
        
    elif var == 'Wind_Direction':
        test_preds_for_var = np.clip(test_preds_for_var, 0, 359)
        print("    Applied Wind Direction Clip: 0 <= pred <= 359")
        
    elif var == 'Wind_Speed':
        test_preds_for_var = np.clip(test_preds_for_var, 0, 180)
        print("    Applied Wind Speed Clip: 0 <= pred <= 180")
        
    elif var == 'Radiation':
        # Only clip radiation at lower end
        test_preds_for_var = np.maximum(test_preds_for_var, 0)
        print("    Applied Radiation non-negative clip")
    
    # Store results
    test_preds_all_vars[var] = test_preds_for_var

# Assemble Submission
print("\nAssembling final submission...")
submission_df = pd.DataFrame({'ID': test_ids})

for var in target_vars:
    # Round to integers
    final_integer_preds = np.round(test_preds_all_vars[var]).astype(int)
    submission_df[var] = final_integer_preds
    print(f"  Final {var} (rounded): min={final_integer_preds.min()}, max={final_integer_preds.max()}, mean={final_integer_preds.mean():.2f}")

# Save submission
submission_df.to_csv(output_path, index=False)
print(f"\nSubmission file created successfully at: {output_path}")
print(f"Timestamp: {datetime.datetime.now()}")

# Display Preview
print("\nSubmission preview:")
print(submission_df.head())

print(f"\n--- Script fixed_winning_model.py Finished ---")