# The Future of Harveston: Climate Variable Prediction

## Overview
This repository contains our solution for the Data Crunch competition on predicting critical environmental variables for Harveston. Our approach combines ensemble methods and advanced feature engineering to forecast five climate variables: Average Temperature, Radiation, Rain Amount, Wind Speed, and Wind Direction. Our final model achieved a score of 36.8 on the competition leaderboard [Public].

## Repository Structure
The repository is organized as follows:

- **Notebooks_and_Scripts/**: Contains all Python scripts for data processing and modeling
  - `winning_strategy_model_36_87832.py`: Our main prediction model (score: 36.8)
  - `advanced_winning_37_00458.py`: Alternative model implementation (score: 37.0)
  - `01_data_exploration.py`: Initial data exploration script
- **final_submission.csv**: Our competition submission with predictions
- **technical_report.pdf**: Detailed documentation of our approach and findings
- **train.csv**: Training dataset
- **test.csv**: Test dataset

## Requirements
To run our models, you will need the following Python packages:

```
pandas
numpy
xgboost
lightgbm
scikit-learn
```

You can install these dependencies using pip with the following command:

```
pip install pandas numpy xgboost lightgbm scikit-learn
```

## How to Run the Model
Follow these steps to execute our winning model:

1. Clone this repository:
   ```
   git clone https://github.com/mehara-rothila/Data_Crunch_079.git
   cd Data_Crunch_079
   ```

2. Ensure that the datasets (`train.csv` and `test.csv`) are in the repository root directory.

3. Run the winning model:
   ```
   python Notebooks_and_Scripts/winning_strategy_model_36_87832.py
   ```

4. The script will generate predictions and save them as `fixed_winning_model_submission.csv` in the root directory.

## Model Features
Our solution incorporates several innovative approaches to climate variable prediction:

- **Innovative approach to wind direction**: Trigonometric decomposition method for circular data
- **Kingdom-specific calibration**: Custom adjustment factors based on regional climate patterns
- **Ensemble methodology**: Strategic weighting of XGBoost and LightGBM models
- **Temporal feature engineering**: Multi-scale cyclical encoding for calendar variables
- **Advanced geographic features**: Spatial-temporal interaction terms

## Team
Team Name: Xforce  
Team ID: Data_Crunch_079  
University: [University Of Moratuwa]


