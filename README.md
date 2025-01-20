# Areca Nut Price Prediction

## Project Overview
This project focuses on predicting areca nut prices using historical data and the SARIMA (Seasonal Autoregressive Integrated Moving Average) model. The goal is to capture seasonal and trend components in price patterns to assist farmers and traders in making well-informed decisions.

## Key Features
- **Data Preparation:** Historical price data was resampled to monthly intervals and transformed for stationarity through differencing.
- **SARIMA Modeling:** The SARIMA model was employed to identify and forecast seasonal trends in areca nut prices.
- **Forecast Accuracy:** The model demonstrated strong alignment between forecasted prices and actual test data, providing valuable insights for planning price-sensitive agricultural activities.

## Use Case
This tool serves as a decision-making aid for:
- Farmers planning crop sales.
- Traders assessing market trends.
- Policymakers devising agricultural support strategies.

## How It Works
1. **Data Collection:** Historical areca nut price data was gathered.
2. **Resampling & Preprocessing:** Data was resampled to monthly intervals, and stationarity was ensured through differencing.
3. **Model Training:** A SARIMA model was trained on the prepared data to capture seasonal and trend components.
4. **Price Forecasting:** The trained model was used to predict future prices, demonstrating high accuracy when compared to test data.

## Results
The SARIMA model successfully identified seasonal trends and provided accurate price forecasts, proving its efficacy as a forecasting tool for the agricultural sector.

## Future Enhancements
- Expanding the dataset to include additional years and regions for improved accuracy.
- Integrating external factors like weather conditions or market demand into the model.
- Developing a user-friendly interface for real-time price forecasting.

---

Feel free to contribute or provide feedback to improve this project!
