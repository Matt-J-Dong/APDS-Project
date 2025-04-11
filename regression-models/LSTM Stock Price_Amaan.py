# Enhanced Multi-Horizon LSTM Stock Price Prediction
# This script allows configuring the amount of historical data used (5 or 10 years)
# and compares predictions using different feature sets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
from datetime import datetime, timedelta
import os
import math

# Configuration
TICKER = 'AAPL'
DATA_FILE = f'{TICKER}_with_sentiment.csv'
YEARS_OF_DATA = 10  # Set to either 5 or 10 years
SEQUENCE_LENGTH = 15 # Number of previous days to use
PREDICTION_HORIZONS = [1, 3]  # Days to predict into the future
RANDOM_SEED = 42
NUM_EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10  # Early stopping patience

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Define the three feature sets we'll compare
FEATURE_SETS = {
    'price_only': ['close_x'],
    'with_technicals': ['close_x', 'RSI', 'MACD', 'ADX', 'OBV', '+DI', '-DI', 'EMA_12', 'EMA_26'],
    'with_sentiment': ['close_x', 'RSI', 'MACD', 'ADX', 'OBV', '+DI', '-DI', 'EMA_12', 'EMA_26', 
                      'positive', 'negative', 'neutral'],
    'price_sentiment': ['close_x', 'positive', 'negative', 'neutral']
}

def load_data(file_path, years=YEARS_OF_DATA):
    """Load the data file and filter to the specified number of years."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['Date_x'] = pd.to_datetime(df['Date_x'])
    
    # Filter to only include data from the last X years
    cutoff_date = datetime.now() - timedelta(days=365 * years)
    df = df[df['Date_x'] >= cutoff_date]
    
    # Sort by date
    df = df.sort_values('Date_x').reset_index(drop=True)
    
    print(f"Data filtered to last {years} years: {df['Date_x'].min().strftime('%Y-%m-%d')} to {df['Date_x'].max().strftime('%Y-%m-%d')}")
    print(f"Total data points: {len(df)}")
    
    # Fill any missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def prepare_multi_horizon_data(df, features, horizons=PREDICTION_HORIZONS, seq_length=SEQUENCE_LENGTH):
    """Prepare data for multi-horizon prediction with the specified feature set."""
    # Ensure features exist in the dataframe
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe columns: {df.columns.tolist()}")
    
    # Extract features
    feature_data = df[features].values
    
    # Target is always closing price
    close_prices = df['close_x'].values.reshape(-1, 1)
    
    # Scale the feature data
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(feature_data)
    
    # Scale the target data (closing prices)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = price_scaler.fit_transform(close_prices)
    
    # Create sequences and targets for each horizon
    X = []
    Y = {horizon: [] for horizon in horizons}
    
    for i in range(len(scaled_features) - seq_length - max(horizons)):
        # Input sequence of features
        X.append(scaled_features[i:i+seq_length])
        
        # Target for each horizon (always closing price)
        for horizon in horizons:
            Y[horizon].append(scaled_prices[i+seq_length+horizon-1, 0])
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = {horizon: np.array(y) for horizon, y in Y.items()}
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    
    Y_train = {horizon: y[:train_size] for horizon, y in Y.items()}
    Y_test = {horizon: y[train_size:] for horizon, y in Y.items()}
    
    # Get dates for test set
    test_dates = {}
    test_prices = {}
    
    for horizon in horizons:
        # Dates for this horizon's predictions
        test_dates[horizon] = df['Date_x'].iloc[train_size+seq_length+horizon-1:train_size+seq_length+horizon-1+len(X_test)]
        # Actual prices for this horizon
        test_prices[horizon] = df['close_x'].iloc[train_size+seq_length+horizon-1:train_size+seq_length+horizon-1+len(X_test)]
    
    # Check shapes for debugging
    print(f"Feature set: {features}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, Y_train, Y_test, test_dates, test_prices, price_scaler, feature_scaler, df

def build_multi_output_model(input_shape, num_outputs):
    """Build an LSTM model with multiple outputs for different horizons."""
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output layer with multiple outputs (one for each horizon)
    model.add(Dense(units=num_outputs))
    
    # Compile
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    model.summary()
    return model

def train_model(model, X_train, Y_train, X_test, Y_test, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE):
    """Train the LSTM model with early stopping."""
    # Combine Y_train values into a single array for multi-output training
    y_train_combined = np.column_stack([Y_train[horizon] for horizon in sorted(Y_train.keys())])
    y_test_combined = np.column_stack([Y_test[horizon] for horizon in sorted(Y_test.keys())])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        X_train, y_train_combined,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test_combined),
        callbacks=callbacks,
        verbose=1
    )

    # Plot training & validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss (MSE)', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)', color='orange')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{TICKER}_{YEARS_OF_DATA}yr_{len(Y_train)}_loss_curve.png')
    plt.close()

    
    return model, history


def evaluate_model(model, X_test, Y_test, price_scaler, test_prices, horizons, num_predictors):
    """Evaluate the model's performance for each prediction horizon, including adjusted R²."""
    # Make predictions for all horizons at once
    y_pred_scaled = model.predict(X_test)

    results = {}

    for i, horizon in enumerate(sorted(horizons)):
        # Extract predictions for this horizon
        horizon_pred_scaled = y_pred_scaled[:, i].reshape(-1, 1)

        # Inverse transform to get actual prices
        horizon_pred_reshaped = np.zeros((len(horizon_pred_scaled), 1))
        horizon_pred_reshaped[:, 0] = horizon_pred_scaled.flatten()
        horizon_pred = price_scaler.inverse_transform(horizon_pred_reshaped).flatten()

        # Get actual prices for this horizon
        actual_prices = test_prices[horizon].values

        # Calculate metrics
        mse = mean_squared_error(actual_prices, horizon_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(actual_prices, horizon_pred)
        r2 = r2_score(actual_prices, horizon_pred)
        mape = np.mean(np.abs((actual_prices - horizon_pred) / actual_prices)) * 100

        # Compute adjusted R²
        n = len(actual_prices)  # number of test samples
        p = num_predictors  # number of predictors/features used for this model
        adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

        results[horizon] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'predictions': horizon_pred
        }

    return results


def plot_future_comparison(df, future_predictions, horizons):
    """Plot the future price predictions from all models."""
    # Get historical prices for context
    historical_dates = df['Date_x'].iloc[-90:]  # Last 90 days
    historical_prices = df['close_x'].iloc[-90:]

    # Plot
    plt.figure(figsize=(16, 8))

    # Plot historical prices
    plt.plot(historical_dates, historical_prices, label='Historical Price', color='blue', linewidth=2)

    # Updated colors and markers for each feature set (4 values now)
    colors = ['red', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'd']

    # Create future dates
    last_date = df['Date_x'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=h) for h in sorted(horizons)]

    # Add the current price point to all lines
    for i, feature_set in enumerate(FEATURE_SETS.keys()):
        # Start with current price
        dates = [last_date]
        prices = [df['close_x'].iloc[-1]]

        # Add each horizon's prediction
        for horizon in sorted(horizons):
            pred = future_predictions[feature_set][horizon]
            dates.append(pred['prediction_date'])
            prices.append(pred['predicted_price'])

        # Plot this feature set's predictions
        plt.plot(dates, prices, label=f'{feature_set.replace("_", " ").title()} Prediction',
                 color=colors[i], linestyle='--', marker=markers[i])

        # Add price labels to the last point
        plt.annotate(f"${prices[-1]:.2f}",
                     (dates[-1], prices[-1]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     color=colors[i])

    plt.title(f'{TICKER} Future Price Predictions ({YEARS_OF_DATA}-Year Training Data)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{TICKER}_{YEARS_OF_DATA}yr_future_predictions.png')
    plt.close()


def predict_future_with_all_models(df, models, feature_scalers, price_scaler, horizons):
    """Predict future prices using all three models and compare."""
    print("\nPredicting future prices with all models...")
    
    future_predictions = {}
    
    # For each feature set and corresponding model
    for feature_set, model in models.items():
        features = FEATURE_SETS[feature_set]
        feature_scaler = feature_scalers[feature_set]
        
        # Get the last sequence of data for these features
        last_sequence = df[features].values[-SEQUENCE_LENGTH:, :]
        
        # Scale the data
        last_sequence_scaled = feature_scaler.transform(last_sequence)
        
        # Reshape for LSTM input [samples, time steps, features]
        X_pred = last_sequence_scaled.reshape(1, SEQUENCE_LENGTH, len(features))
        
        # Make predictions for all horizons
        predictions_scaled = model.predict(X_pred)[0]
        
        # Process predictions for each horizon
        future_predictions[feature_set] = {}
        
        for i, horizon in enumerate(sorted(horizons)):
            # Extract and reshape prediction for this horizon
            horizon_pred_scaled = predictions_scaled[i].reshape(1, 1)
            
            # Inverse transform
            # We need to prepare a properly shaped array for inverse_transform
            horizon_pred_reshaped = np.zeros((1, 1))
            horizon_pred_reshaped[0, 0] = horizon_pred_scaled[0, 0]
            
            predicted_price = price_scaler.inverse_transform(horizon_pred_reshaped)[0, 0]
            
            # Current price
            current_price = df['close_x'].iloc[-1]
            
            # Calculate change
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            
            direction = "UP" if change > 0 else "DOWN"
            
            # Calculate future date
            last_date = df['Date_x'].iloc[-1]
            future_date = last_date + pd.Timedelta(days=horizon)
            
            # Store prediction
            future_predictions[feature_set][horizon] = {
                'current_date': last_date,
                'prediction_date': future_date,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'change': change,
                'change_pct': change_pct,
                'direction': direction
            }
    
    return future_predictions

def plot_future_comparison(df, future_predictions, horizons):
    """Plot the future price predictions from all models."""
    # Get historical prices for context
    historical_dates = df['Date_x'].iloc[-90:]  # Last 90 days
    historical_prices = df['close_x'].iloc[-90:]
    
    # Plot
    plt.figure(figsize=(16, 8))
    
    # Plot historical prices
    plt.plot(historical_dates, historical_prices, label='Historical Price', color='blue', linewidth=2)
    
    # Colors and markers for each feature set
    colors = ['red', 'green', 'purple']
    markers = ['o', 's', '^']
    
    # Create future dates
    last_date = df['Date_x'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=h) for h in sorted(horizons)]
    
    # Add the current price point to all lines
    for i, feature_set in enumerate(FEATURE_SETS.keys()):
        # Start with current price
        dates = [last_date]
        prices = [df['close_x'].iloc[-1]]
        
        # Add each horizon's prediction
        for horizon in sorted(horizons):
            pred = future_predictions[feature_set][horizon]
            dates.append(pred['prediction_date'])
            prices.append(pred['predicted_price'])
        
        # Plot this feature set's predictions
        plt.plot(dates, prices, label=f'{feature_set.replace("_", " ").title()} Prediction', 
                 color=colors[i], linestyle='--', marker=markers[i])
        
        # Add price labels to the last point
        plt.annotate(f"${prices[-1]:.2f}", 
                    (dates[-1], prices[-1]),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    color=colors[i])
    
    plt.title(f'{TICKER} Future Price Predictions ({YEARS_OF_DATA}-Year Training Data)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{TICKER}_{YEARS_OF_DATA}yr_future_predictions.png')
    plt.close()

def main():
    """Main function to run the enhanced multi-horizon LSTM prediction pipeline."""
    print(f"Running enhanced multi-horizon LSTM price prediction for {TICKER} using {YEARS_OF_DATA} years of data...")
    
    # Load data with specified years
    df = load_data(DATA_FILE, years=YEARS_OF_DATA)
    
    # Initialize dictionaries to store results
    models = {}
    histories = {}
    all_results = {}
    feature_scalers = {}
    
    # For each feature set, train and evaluate a separate model
    for feature_set, features in FEATURE_SETS.items():
        print(f"\n{'='*50}")
        print(f"Processing feature set: {feature_set}")
        print(f"Features: {features}")
        print(f"{'='*50}\n")
        
        # Prepare data
        X_train, X_test, Y_train, Y_test, test_dates, test_prices, price_scaler, feature_scaler, _ = prepare_multi_horizon_data(
            df, features, horizons=PREDICTION_HORIZONS, seq_length=SEQUENCE_LENGTH
        )
        
        # Save scalers
        feature_scalers[feature_set] = feature_scaler
        
        # Build model
        model = build_multi_output_model((SEQUENCE_LENGTH, len(features)), len(PREDICTION_HORIZONS))
        
        # Train model
        model, history = train_model(model, X_train, Y_train, X_test, Y_test, patience=PATIENCE)
        
        # Evaluate model
        results = evaluate_model(model, X_test, Y_test, price_scaler, test_prices, PREDICTION_HORIZONS,
                                 num_predictors=len(features))

        # Store results
        models[feature_set] = model
        histories[feature_set] = history
        all_results[feature_set] = results
        
        # Print summary of results
        print(f"\nResults summary for {feature_set}:")
        for horizon in sorted(PREDICTION_HORIZONS):
            print(f"{horizon}-Day Horizon:")
            print(f"  RMSE: ${results[horizon]['rmse']:.2f}")
            print(f"  MAE: ${results[horizon]['mae']:.2f}")
            print(f"  MAPE: {results[horizon]['mape']:.2f}%")
            print(f"  R²: {results[horizon]['r2']:.4f}")
            print(f"  Adjusted R²: {results[horizon]['adjusted_r2']:.4f}")

        # Save individual model
        model.save(f'{TICKER}_{YEARS_OF_DATA}yr_{feature_set}_model.h5')
    
    # Plot comparison results
    plot_comparison_results(PREDICTION_HORIZONS, all_results, test_dates, test_prices)
    
    # Predict future prices with all models
    future_predictions = predict_future_with_all_models(df, models, feature_scalers, price_scaler, PREDICTION_HORIZONS)
    
    # Plot future predictions comparison
    plot_future_comparison(df, future_predictions, PREDICTION_HORIZONS)
    
    # Save comprehensive results
    with open(f'{TICKER}_{YEARS_OF_DATA}yr_enhanced_predictions.txt', 'w') as f:
        f.write(f"Enhanced Multi-Horizon LSTM Stock Price Prediction for {TICKER}\n")
        f.write(f"Using {YEARS_OF_DATA} years of historical data\n")
        f.write("="*70 + "\n\n")
        
        # Current price info
        current_price = df['close_x'].iloc[-1]
        current_date = df['Date_x'].iloc[-1]
        f.write(f"Current Price (as of {current_date.strftime('%Y-%m-%d')}): ${current_price:.2f}\n\n")
        
        # Feature set comparison
        f.write("FEATURE SET COMPARISON\n")
        f.write("-"*70 + "\n\n")
        
        # Write comparison table for each horizon
        for horizon in sorted(PREDICTION_HORIZONS):
            f.write(f"Horizon: {horizon} Days\n")
            f.write("-"*50 + "\n")
            
            # Create a comparison table
            for metric in ['rmse', 'mae', 'mape', 'r2']:
                metric_name = {'rmse': 'RMSE ($)', 'mae': 'MAE ($)', 'mape': 'MAPE (%)', 'r2': 'R² Score'}[metric]
                baseline = all_results['price_only'][horizon][metric]
                
                f.write(f"{metric_name}:\n")
                
                for feature_set in FEATURE_SETS.keys():
                    value = all_results[feature_set][horizon][metric]
                    
                    if feature_set == 'price_only':
                        f.write(f"  {feature_set.replace('_', ' ').title()}: {value:.4f}\n")
                    else:
                        if metric != 'r2':  # For RMSE, MAE, MAPE lower is better
                            improvement = ((baseline - value) / baseline) * 100
                            f.write(f"  {feature_set.replace('_', ' ').title()}: {value:.4f} ({improvement:.2f}% improvement)\n")
                        else:  # For R², higher is better
                            improvement = ((value - baseline) / abs(baseline)) * 100
                            f.write(f"  {feature_set.replace('_', ' ').title()}: {value:.4f} ({improvement:.2f}% improvement)\n")
                
                f.write("\n")
            
            f.write("\n")
        
        # Future predictions
        f.write("FUTURE PREDICTIONS\n")
        f.write("-"*70 + "\n\n")
        
        for horizon in sorted(PREDICTION_HORIZONS):
            f.write(f"{horizon}-Day Horizon ({future_predictions['price_only'][horizon]['prediction_date'].strftime('%Y-%m-%d')}):\n")
            
            for feature_set in FEATURE_SETS.keys():
                pred = future_predictions[feature_set][horizon]
                f.write(f"  {feature_set.replace('_', ' ').title()}: ${pred['predicted_price']:.2f} ({pred['change_pct']:.2f}%, {pred['direction']})\n")
            
            f.write("\n")
        
        # Dataset information
        f.write("\nDATASET INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Data range: {df['Date_x'].min().strftime('%Y-%m-%d')} to {df['Date_x'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"Total data points: {len(df)}\n")
        f.write(f"Training data: {int(len(df)*0.8)} points ({80}%)\n")
        f.write(f"Testing data: {len(df) - int(len(df)*0.8)} points ({20}%)\n")
    
    print(f"\nEnhanced prediction complete! Results saved to {TICKER}_{YEARS_OF_DATA}yr_enhanced_predictions.txt")
    
    return future_predictions, all_results

if __name__ == "__main__":
    main()