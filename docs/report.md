# Electricity Consumption Data Analysis

## 1. Dataset Overview

**Name**: Electricity Load Diagrams 2011-2014  
**Source**: [Kaggle - Michael R Looney](https://www.kaggle.com/datasets/michaelrlooney/electricity-load-diagrams-2011-2014)  
**License**: CC BY 4.0

- **Time Range**: 2011-01-01 00:15:00 to 2015-01-01 00:00:00  
- **Sampling Frequency**: 15-minute intervals  
- **Number of Meters**: 370 (columns `MT_001` through `MT_370`)  
- **Data Shape**: 140,256 rows x 370 columns (about 700+ MB)  
- **Units**: kWh (kilowatt-hours)

## 2. Initial Observations

1. **Missing Values**: None found (all columns have 0 missing).  
2. **All-Zero Rows**: Some timestamps may show zero consumption for certain meters. Need to investigate if this is normal (e.g., meter off) or data anomaly.  
3. **Seasonality**: Visual inspection suggests possible higher usage in certain months/years, but we need further analysis (daily or weekly aggregates).  
4. **Correlation**: A quick correlation among a few meters (`MT_001` to `MT_005`) shows moderate to high correlation, indicating similar usage patterns.

## 3. Exploration Highlights

- **Time Series Plots**:  
  - Notable spikes around [year or time period].  
  - Some meters remain at near-zero for extended periods.  

- **Daily Resampling**:  
  - Aggregating data to daily sums reveals clearer long-term trends.  
  - Potential seasonal fluctuations (higher in winter/summer?).

## 4. Potential Anomalies

- **Zero Consumption** for entire periods: Could be a meter malfunction or an unoccupied building.  
- **Sudden Spikes**: Large jumps in consumption might indicate data errors or real anomalies (e.g., equipment turning on unexpectedly).

## 5. Next Steps

1. **Data Preprocessing**  
   - Possibly remove or impute extended zero values if deemed invalid.  
   - Resample to daily or hourly intervals for simpler modeling.

2. **Feature Engineering**  
   - Rolling window statistics (mean, std, min, max) to capture local trends.  
   - Time-based features (hour of day, day of week, month).

3. **Anomaly Detection Approaches**  
   - **Statistical**: Rolling z-score, thresholding.  
   - **Machine Learning**: Isolation Forest, Local Outlier Factor.  
   - **Deep Learning**: Autoencoders, LSTM-based forecasting and anomaly scoring.

## 6. Conclusion

The dataset is large and shows strong patterns of daily/seasonal behavior. Further preprocessing and advanced modeling (e.g., autoencoders) may help detect consumption anomalies effectively. Future work involves splitting the data by time (train on 2011-2013, test on 2014), engineering features, and selecting an anomaly detection approach.

---
