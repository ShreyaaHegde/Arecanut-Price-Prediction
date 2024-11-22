import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Streamlit App Interface
st.title("Arecanut Price Analysis and Forecasting")

# Step 1: File Upload
st.header("1. Upload the Arecanut Price Data (Excel)")
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.write("Dataset Overview:")
    st.dataframe(data)

    # Ensure 'Date' column is recognized as datetime format
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Avg_Price'] = pd.to_numeric(data['Avg_Price'], errors='coerce')

        st.header("2. Data Visualization")
        
        # Check if 'Branch' column exists
        if 'Branch' in data.columns:
            
            # 1. Line Plot - Trend of Avg Price Over Time
            st.subheader("Trend of Arecanut Average Prices Over Time")
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='Date', y='Avg_Price', hue='Branch', data=data, palette='Set2')
            plt.title('Trend of Average Prices Over Time')
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # 2. Box Plot - Distribution of Prices by Branch
            st.subheader("Distribution of Average Prices by Branch")
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Branch', y='Avg_Price', data=data, palette='Set3')
            plt.title('Distribution of Average Prices by Branch')
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # 3. Heatmap - Correlation between numerical variables
            st.subheader("Correlation Heatmap of Numerical Features")
            corr = data[['Avg_Price', 'Year', 'Month']].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Heatmap')
            st.pyplot(plt)

        # ADF Test for stationarity
        st.subheader("ADF Test for Stationarity and Differencing")
        time_series_data = data['Avg_Price'].dropna()

        # Plot original time series data before stationarity
        st.subheader("Original Avg Price Time Series (Before Differencing)")
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Avg_Price'], label='Original Avg Price', color='blue')
        plt.title("Original Avg Price Time Series")
        plt.xlabel("Date")
        plt.ylabel("Avg Price")
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Perform the ADF test on the original series
        adf_result = adfuller(time_series_data)
        st.write(f'ADF Statistic: {adf_result[0]}')
        st.write(f'p-value: {adf_result[1]}')
        st.write('Critical Values:')
        for key, value in adf_result[4].items():
            st.write(f'   {key}: {value}')

        # Perform seasonal differencing
        data['Avg_Price_seasonal_diff'] = data['Avg_Price'] - data['Avg_Price'].shift(12)

        # Perform the ADF test on the seasonally differenced series
        st.subheader("Seasonally Differenced Avg Price (After Differencing)")
        plt.figure(figsize=(10, 6))
        data['Avg_Price_seasonal_diff'].dropna().plot()
        plt.title("Seasonally Differenced Avg Price Time Series")
        plt.xlabel("Date")
        plt.ylabel("Seasonally Differenced Avg Price")
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # ADF Test for the differenced data
        seasonally_diff_data = data['Avg_Price_seasonal_diff'].dropna()
        adf_result_diff = adfuller(seasonally_diff_data)
        st.write(f'ADF Statistic (After Differencing): {adf_result_diff[0]}')
        st.write(f'p-value (After Differencing): {adf_result_diff[1]}')
        st.write('Critical Values (After Differencing):')
        for key, value in adf_result_diff[4].items():
            st.write(f'   {key}: {value}')

        # SARIMA Model Forecasting
        st.header("3. Forecasting with SARIMA Model")
        branches = data['Branch'].unique()
        selected_branch = st.selectbox("Select Branch", branches)
        selected_year = st.number_input("Select Future Year", min_value=2024, value=2024, step=1)
        selected_month = st.number_input("Select Future Month (1-12)", min_value=1, max_value=12, value=1)

        # Filter data for the selected branch
        branch_data = data[data['Branch'] == selected_branch]

        if not branch_data.empty:
            # Prepare the data for SARIMA
            branch_data.set_index('Date', inplace=True)
            monthly_data = branch_data.resample('M').mean(numeric_only=True)

            # Split data into training and testing sets
            split_point = int(len(monthly_data) * 0.8)  # 80% training, 20% testing
            train_data = monthly_data.iloc[:split_point]
            test_data = monthly_data.iloc[split_point:]

            # Fit the SARIMA model on the training data
            model = SARIMAX(train_data['Avg_Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            results = model.fit()

            # Predict historical test period (previous years)
            pred_start = test_data.index[0]
            pred_end = test_data.index[-1]
            past_predictions = results.get_prediction(start=pred_start, end=pred_end)
            predicted_past_values = past_predictions.predicted_mean

            # Historical predictions vs actual prices
            historical_df = pd.DataFrame({
                'Date': test_data.index, 
                'Actual Price': test_data['Avg_Price'], 
                'Predicted Price': predicted_past_values
            })

            # Make future predictions
            forecast_steps = 5  # Number of months for future prediction
            future_dates = pd.date_range(start=f"{selected_year}-{selected_month:02d}-01", periods=forecast_steps, freq='M')
            future_forecast = results.get_forecast(steps=forecast_steps)
            future_forecast_values = future_forecast.predicted_mean

            # Future forecasted values
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Avg Price': future_forecast_values})

            # Display historical predictions and future forecast
            st.subheader("Historical Predictions vs Actual Prices:")
            st.write(historical_df)

            st.subheader(f"Predicted Prices for Arecanut (Future from {selected_year}-{selected_month}):")
            st.write(future_df)

            # Plot the historical data, training, testing, predicted values, and future forecasts
            plt.figure(figsize=(10, 6))
            plt.plot(monthly_data.index, monthly_data['Avg_Price'], label='Historical Prices', color='blue')
            plt.plot(train_data.index, train_data['Avg_Price'], label='Training Data', color='green')
            plt.plot(test_data.index, test_data['Avg_Price'], label='Test Data', color='red')
            plt.plot(historical_df['Date'], historical_df['Predicted Price'], label='Predicted Prices (Past)', color='purple')
            plt.plot(future_df['Date'], future_df['Predicted Avg Price'], label='Forecasted Prices (Future)', color='orange')
            plt.title(f'Arecanut Price Forecast for {selected_branch}')
            plt.xlabel('Date')
            plt.ylabel('Average Price')
            plt.legend()
            st.pyplot(plt)
        else:
            st.error(f"No data found for the branch: {selected_branch}")

else:
    st.warning("Please upload an Excel file.")
