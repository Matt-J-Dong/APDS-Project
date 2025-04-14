# Enhanced Multi-Horizon LSTM Stock Price Prediction with Optimization
# Optimized with: Numba, GPU Acceleration, and Multiprocessing
# This script allows configuring the amount of historical data used (5 or 10 years)
# and compares predictions using different feature sets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta
import os
import math
import time
from numba import jit, prange, float64, int64, boolean
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import warnings
import gc  # Garbage collection
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread-safe plotting
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
TICKER = 'AAPL'
DATA_FILE = f'{TICKER}_with_sentiment.csv'
YEARS_OF_DATA = 10
SEQUENCE_LENGTH = 15
PREDICTION_HORIZONS = [1, 3]
RANDOM_SEED = 42
NUM_EPOCHS = 100
BATCH_SIZE = 64  # Increased from 32
PATIENCE = 10

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s) for acceleration")
    except RuntimeError as e:
        print(f"GPU error: {e}")
else:
    print("No GPU found, using CPU")

# Clear any existing models/sessions
tf.keras.backend.clear_session()

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Define the feature sets to compare
FEATURE_SETS = {
    'price_only': ['close_x'],
    'with_technicals': ['close_x', 'RSI', 'MACD', 'ADX', 'OBV', '+DI', '-DI', 'EMA_12', 'EMA_26'],
    'with_sentiment': ['close_x', 'RSI', 'MACD', 'ADX', 'OBV', '+DI', '-DI', 'EMA_12', 'EMA_26',
                       'positive', 'negative', 'neutral'],
    'price_sentiment': ['close_x', 'positive', 'negative', 'neutral']
}


def load_data(file_path, years=YEARS_OF_DATA):
    """Load the data file, filter to the specified number of years, sort by date, and fill missing values."""
    start_time = time.time()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df['Date_x'] = pd.to_datetime(df['Date_x'])

    # Filter to only include data from the last X years
    cutoff_date = datetime.now() - timedelta(days=365 * years)
    df = df[df['Date_x'] >= cutoff_date]

    # Sort by date
    df = df.sort_values('Date_x').reset_index(drop=True)

    print(
        f"Data filtered to last {years} years: {df['Date_x'].min().strftime('%Y-%m-%d')} to {df['Date_x'].max().strftime('%Y-%m-%d')}")
    print(f"Total data points: {len(df)}")

    # Use ffill() and bfill() to avoid future warnings
    df = df.ffill().bfill()
    
    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    return df


@jit(nopython=True, parallel=True)
def create_sequences_numba(scaled_features, scaled_prices, seq_length, max_horizon):
    """Use Numba to accelerate sequence creation for LSTM."""
    n_samples = len(scaled_features) - seq_length - max_horizon
    n_features = scaled_features.shape[1]
    
    # Pre-allocate arrays
    X = np.zeros((n_samples, seq_length, n_features))
    Y_dict = {}
    for h in range(1, max_horizon + 1):
        Y_dict[h] = np.zeros(n_samples)
    
    # Create sequences in parallel
    for i in prange(n_samples):
        # Feature sequences
        for j in range(seq_length):
            for k in range(n_features):
                X[i, j, k] = scaled_features[i + j, k]
        
        # Target values for each horizon
        for h in range(1, max_horizon + 1):
            if i + seq_length + h - 1 < len(scaled_prices):
                Y_dict[h][i] = scaled_prices[i + seq_length + h - 1, 0]
    
    return X, Y_dict


def prepare_data_chunk(args):
    """Process a chunk of data for multiprocessing."""
    chunk_idx, df_chunk, features, horizons, seq_length, feature_scaler, price_scaler = args
    
    # Extract feature data and price data
    feature_data = df_chunk[features].values
    close_prices = df_chunk['close_x'].values.reshape(-1, 1)
    
    # Scale the data (use the passed scalers)
    scaled_features = feature_scaler.transform(feature_data)
    scaled_prices = price_scaler.transform(close_prices)
    
    # Create sequences
    max_horizon = max(horizons)
    X_chunk, Y_dict = create_sequences_numba(scaled_features, scaled_prices, seq_length, max_horizon)
    
    # Filter out sequences that don't have enough future data
    valid_length = min(len(X_chunk), len(Y_dict[max(horizons)]))
    X_chunk = X_chunk[:valid_length]
    Y_chunk = {h: Y_dict[h][:valid_length] for h in horizons}
    
    return chunk_idx, X_chunk, Y_chunk


def prepare_multi_horizon_data(df, features, horizons=PREDICTION_HORIZONS, seq_length=SEQUENCE_LENGTH):
    """Prepare data for multi-horizon prediction with efficient processing."""
    start_time = time.time()
    
    # Validate features
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe columns: {df.columns.tolist()}")

    # Extract data efficiently
    feature_data = df[features].values
    close_prices = df['close_x'].values.reshape(-1, 1)

    # Scale features and target
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(feature_data)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = price_scaler.fit_transform(close_prices)

    # Pre-allocate arrays for better memory efficiency
    n_samples = len(scaled_features) - seq_length - max(horizons)
    X = np.zeros((n_samples, seq_length, len(features)))
    Y = {horizon: np.zeros(n_samples) for horizon in horizons}

    # Create sequences efficiently
    for i in range(n_samples):
        X[i] = scaled_features[i:i + seq_length]
        for horizon in horizons:
            Y[horizon][i] = scaled_prices[i + seq_length + horizon - 1, 0]

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train = {horizon: y[:train_size] for horizon, y in Y.items()}
    Y_test = {horizon: y[train_size:] for horizon, y in Y.items()}

    # Get dates and prices for test set
    test_dates = {}
    test_prices = {}
    for horizon in horizons:
        idx_start = train_size + seq_length + horizon - 1
        idx_end = idx_start + len(X_test)
        test_dates[horizon] = df['Date_x'].iloc[idx_start:idx_end]
        test_prices[horizon] = df['close_x'].iloc[idx_start:idx_end]

    print(f"Feature set: {features}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")

    return X_train, X_test, Y_train, Y_test, test_dates, test_prices, price_scaler, feature_scaler, df


def build_multi_output_model(input_shape, num_outputs):
    """Build an LSTM model with multiple outputs for different horizons."""
    model = Sequential()
    
    # Match the original architecture
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_outputs))

    # Use efficient optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    model.summary()
    return model


def train_model(model, X_train, Y_train, X_test, Y_test, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                patience=PATIENCE, output_dir="."):
    """Train the LSTM model with early stopping and optimized parameters."""
    start_time = time.time()
    
    y_train_combined = np.column_stack([Y_train[horizon] for horizon in sorted(Y_train.keys())])
    y_test_combined = np.column_stack([Y_test[horizon] for horizon in sorted(Y_test.keys())])

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # Train with optimized batch size
    history = model.fit(
        X_train, y_train_combined,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test_combined),
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss (MSE)', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)', color='orange')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_{len(Y_train)}_loss_curve.png'))
    plt.close()

    print(f"Model training completed in {time.time() - start_time:.2f} seconds")
    return model, history


@jit(nopython=True)
def calculate_direction_accuracy(actual_prices, predicted_prices):
    """Calculate the accuracy of predicted price direction with Numba acceleration."""
    n = len(actual_prices) - 1
    correct_count = 0
    
    for i in range(1, len(actual_prices)):
        actual_up = actual_prices[i] > actual_prices[i-1]
        pred_up = predicted_prices[i] > predicted_prices[i-1]
        
        if actual_up == pred_up:
            correct_count += 1
            
    return (correct_count / n) * 100


@jit(nopython=True)
def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics with Numba acceleration."""
    n = len(actual)
    
    # Calculate MSE
    mse = 0.0
    for i in range(n):
        diff = actual[i] - predicted[i]
        mse += diff * diff
    mse /= n
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    # Calculate MAE
    mae = 0.0
    for i in range(n):
        mae += np.abs(actual[i] - predicted[i])
    mae /= n
    
    # Calculate MAPE
    mape = 0.0
    for i in range(n):
        if actual[i] != 0:
            mape += np.abs((actual[i] - predicted[i]) / actual[i])
    mape = (mape / n) * 100
    
    return rmse, mae, mape


def calculate_r2(actual, predicted):
    """Calculate R² score (not Numba-optimized because it's more complex)."""
    return r2_score(actual, predicted)


def evaluate_model_on_training_data(model, X_train, Y_train, price_scaler, horizons, num_predictors):
    """Evaluate training performance for each horizon with optimized calculations."""
    start_time = time.time()
    
    # Make predictions
    y_train_pred_scaled = model.predict(X_train, batch_size=BATCH_SIZE * 2, verbose=0)
    
    results = {}
    for i, horizon in enumerate(sorted(horizons)):
        # Get predictions for this horizon
        horizon_pred_scaled = y_train_pred_scaled[:, i].reshape(-1, 1)
        horizon_pred_reshaped = np.zeros((len(horizon_pred_scaled), 1))
        horizon_pred_reshaped[:, 0] = horizon_pred_scaled.flatten()
        
        # Inverse transform to get original scale
        horizon_pred = price_scaler.inverse_transform(horizon_pred_reshaped).flatten()
        actual_train = price_scaler.inverse_transform(Y_train[horizon].reshape(-1, 1)).flatten()
        
        # Calculate metrics with optimized functions
        rmse, mae, mape = calculate_metrics(actual_train, horizon_pred)
        r2 = calculate_r2(actual_train, horizon_pred)
        direction_acc = calculate_direction_accuracy(actual_train, horizon_pred)
        
        # Calculate adjusted R²
        n = len(actual_train)
        p = num_predictors
        adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
        
        results[horizon] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'direction_accuracy': direction_acc,
            'predictions': horizon_pred
        }
    
    print(f"Training evaluation completed in {time.time() - start_time:.2f} seconds")
    return results


def evaluate_model(model, X_test, Y_test, price_scaler, test_prices, horizons, num_predictors):
    """Evaluate the model's performance on the test set with optimized calculations."""
    start_time = time.time()
    
    # Make predictions
    y_pred_scaled = model.predict(X_test, batch_size=BATCH_SIZE * 2, verbose=0)
    
    results = {}
    for i, horizon in enumerate(sorted(horizons)):
        # Get predictions for this horizon
        horizon_pred_scaled = y_pred_scaled[:, i].reshape(-1, 1)
        horizon_pred_reshaped = np.zeros((len(horizon_pred_scaled), 1))
        horizon_pred_reshaped[:, 0] = horizon_pred_scaled.flatten()
        
        # Inverse transform to get original scale
        horizon_pred = price_scaler.inverse_transform(horizon_pred_reshaped).flatten()
        actual_prices_array = test_prices[horizon].values
        
        # Ensure same length for comparison
        n = min(len(horizon_pred), len(actual_prices_array))
        actual_prices_array = actual_prices_array[:n]
        horizon_pred = horizon_pred[:n]
        
        # Calculate metrics with optimized functions
        rmse, mae, mape = calculate_metrics(actual_prices_array, horizon_pred)
        r2 = calculate_r2(actual_prices_array, horizon_pred)
        direction_acc = calculate_direction_accuracy(actual_prices_array, horizon_pred)
        
        # Calculate adjusted R²
        n = len(actual_prices_array)
        p = num_predictors
        adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
        
        results[horizon] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'direction_accuracy': direction_acc,
            'predictions': horizon_pred
        }
    
    print(f"Test evaluation completed in {time.time() - start_time:.2f} seconds")
    return results


def plot_comparison_results(horizons, all_results, test_dates, test_prices, output_dir="."):
    """Plot performance comparison between feature sets and save plots to output_dir."""
    start_time = time.time()
    
    metrics = ['rmse', 'mae', 'mape', 'r2', 'direction_accuracy']
    metric_names = ['RMSE ($)', 'MAE ($)', 'MAPE (%)', 'R² Score', 'Direction Accuracy (%)']
    
    for horizon in sorted(horizons):
        # Create two separate figures - one for error metrics and one for accuracy metrics
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
        axes1 = axes1.flatten()
        
        # Create a separate figure for direction accuracy
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Plot error metrics (RMSE, MAE, MAPE, R²)
        for i, (metric, name) in enumerate(zip(metrics[:4], metric_names[:4])):
            metric_values = []
            for feature_set in FEATURE_SETS.keys():
                metric_values.append(all_results[feature_set][horizon][metric])
            df_plot = pd.DataFrame({
                'Feature Set': list(FEATURE_SETS.keys()),
                name: metric_values
            })
            sns.barplot(x='Feature Set', y=name, data=df_plot, ax=axes1[i])
            axes1[i].set_title(f'{name} for {horizon}-Day Horizon')
            axes1[i].grid(True, axis='y')
            if metric != 'r2':
                baseline = metric_values[0]
                for j, value in enumerate(metric_values):
                    if j > 0:
                        improvement = ((baseline - value) / baseline) * 100
                        axes1[i].text(j, value, f"{improvement:.1f}%", ha='center', va='bottom')
            else:
                baseline = metric_values[0]
                for j, value in enumerate(metric_values):
                    if j > 0:
                        improvement = ((value - baseline) / abs(baseline)) * 100
                        axes1[i].text(j, value, f"+{improvement:.1f}%", ha='center', va='bottom')
        
        # Plot direction accuracy
        direction_values = []
        for feature_set in FEATURE_SETS.keys():
            direction_values.append(all_results[feature_set][horizon]['direction_accuracy'])
        
        df_direction = pd.DataFrame({
            'Feature Set': list(FEATURE_SETS.keys()),
            'Direction Accuracy (%)': direction_values
        })
        
        bar_plot = sns.barplot(x='Feature Set', y='Direction Accuracy (%)', data=df_direction, ax=ax2)
        ax2.set_title(f'Direction Accuracy for {horizon}-Day Horizon')
        ax2.grid(True, axis='y')
        
        # Add percentage values on top of the bars
        baseline = direction_values[0]
        for j, value in enumerate(direction_values):
            ax2.text(j, value, f"{value:.1f}%", ha='center', va='bottom')
            if j > 0:
                improvement = ((value - baseline) / baseline) * 100
                ax2.text(j, value/2, f"+{improvement:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        
        ax2.set_ylim(0, max(direction_values) * 1.15)  # Add some space for the labels
        
        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_{horizon}day_metric_comparison.png'))
        fig2.savefig(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_{horizon}day_direction_accuracy.png'))
        plt.close(fig1)
        plt.close(fig2)

    # Price prediction comparison plots
    for horizon in sorted(horizons):
        plt.figure(figsize=(16, 8))
        plt.plot(test_dates[horizon], test_prices[horizon], label='Actual Price', color='blue', linewidth=2)
        colors = ['red', 'green', 'purple', 'orange']
        linestyles = ['--', '-.', ':', '-']
        for i, feature_set in enumerate(FEATURE_SETS.keys()):
            # Ensure the predictions array is the right length
            pred = all_results[feature_set][horizon]['predictions']
            dates = test_dates[horizon]
            prices = test_prices[horizon]
            min_len = min(len(pred), len(dates), len(prices))
            
            plt.plot(dates[:min_len], pred[:min_len],
                     label=f'{feature_set.replace("_", " ").title()}',
                     color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        
        plt.title(f'{TICKER} Stock Price Prediction Comparison ({horizon}-Day Horizon, {YEARS_OF_DATA}-Year Data)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_{horizon}day_prediction_comparison.png'))
        plt.close()
    
    print(f"Comparison plots created in {time.time() - start_time:.2f} seconds")


def predict_future_with_all_models(df, models, feature_scalers, price_scaler, horizons, all_results):
    """Predict future prices using all models with performance optimization."""
    start_time = time.time()
    print("\nPredicting future prices with all models...")
    
    future_predictions = {}
    for feature_set, model in models.items():
        features = FEATURE_SETS[feature_set]
        feature_scaler = feature_scalers[feature_set]
        
        # Get the last sequence
        last_sequence = df[features].values[-SEQUENCE_LENGTH:, :]
        last_sequence_scaled = feature_scaler.transform(last_sequence)
        X_pred = last_sequence_scaled.reshape(1, SEQUENCE_LENGTH, len(features))
        
        # Make prediction
        predictions_scaled = model.predict(X_pred, verbose=0)[0]
        
        future_predictions[feature_set] = {}
        for i, horizon in enumerate(sorted(horizons)):
            # Extract prediction for this horizon
            horizon_pred_scaled = predictions_scaled[i].reshape(1, 1)
            horizon_pred_reshaped = np.zeros((1, 1))
            horizon_pred_reshaped[0, 0] = horizon_pred_scaled[0, 0]
            
            # Convert to original price scale
            predicted_price = price_scaler.inverse_transform(horizon_pred_reshaped)[0, 0]
            current_price = df['close_x'].iloc[-1]
            
            # Calculate change metrics
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            direction = "UP" if change > 0 else "DOWN"
            
            # Get the direction accuracy for this feature set and horizon
            direction_accuracy = all_results[feature_set][horizon]['direction_accuracy']
            
            # Determine confidence level based on direction accuracy
            if direction_accuracy >= 70:
                confidence = "High"
            elif direction_accuracy >= 60:
                confidence = "Medium"
            else:
                confidence = "Low"
                
            last_date = df['Date_x'].iloc[-1]
            future_date = last_date + pd.Timedelta(days=horizon)
            
            future_predictions[feature_set][horizon] = {
                'current_date': last_date,
                'prediction_date': future_date,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'change': change,
                'change_pct': change_pct,
                'direction': direction,
                'direction_accuracy': direction_accuracy,
                'confidence': confidence
            }
    
    print(f"Future predictions completed in {time.time() - start_time:.2f} seconds")
    return future_predictions


def plot_future_comparison(df, future_predictions, horizons, output_dir="."):
    """Plot comparison of future predictions from different feature sets."""
    start_time = time.time()
    
    # Get the last known price and date
    last_price = df['close_x'].iloc[-1]
    last_date = df['Date_x'].iloc[-1]
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot the last 30 days of actual prices
    last_30_days = df.tail(30)
    plt.plot(last_30_days['Date_x'], last_30_days['close_x'], 
             label='Historical', color='black', linewidth=2)
    
    # Colors and markers for different feature sets
    colors = plt.cm.Set3(np.linspace(0, 1, len(future_predictions)))
    markers = ['o', 's', '^', 'D']
    
    # Plot predictions for each feature set
    for (feature_set, predictions), color, marker in zip(future_predictions.items(), colors, markers):
        dates = []
        prices = []
        
        for horizon in sorted(horizons):
            if horizon in predictions:
                pred_date = predictions[horizon]['prediction_date']
                pred_price = predictions[horizon]['predicted_price']
                dates.append(pred_date)
                prices.append(pred_price)
        
        if dates and prices:  # Only plot if we have predictions
            plt.plot(dates, prices, 
                    label=f'{feature_set.replace("_", " ").title()}',
                    color=color, marker=marker, markersize=8, linestyle='--')
            
            # Add price labels to the last point
            plt.annotate(f"${prices[-1]:.2f}", 
                        (dates[-1], prices[-1]),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center',
                        color=color)
    
    plt.title(f'{TICKER} Future Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_future_predictions.png'))
    plt.close()
    
    print(f"Future comparison plot completed in {time.time() - start_time:.2f} seconds")


def process_feature_set(args):
    """Process a single feature set with optimized parallel execution."""
    feature_set, features, df, output_dir = args
    start_time = time.time()

    print(f"\n{'=' * 50}")
    print(f"Processing feature set: {feature_set}")
    print(f"Features: {features}")
    print(f"{'=' * 50}\n")

    X_train, X_test, Y_train, Y_test, test_dates, test_prices, price_scaler, feature_scaler, df = prepare_multi_horizon_data(
        df, features, horizons=PREDICTION_HORIZONS, seq_length=SEQUENCE_LENGTH
    )

    model = build_multi_output_model((SEQUENCE_LENGTH, len(features)), len(PREDICTION_HORIZONS))
    model, history = train_model(model, X_train, Y_train, X_test, Y_test, patience=PATIENCE, output_dir=output_dir)

    results = evaluate_model(model, X_test, Y_test, price_scaler, test_prices, PREDICTION_HORIZONS,
                           num_predictors=len(features))

    model.save(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_{feature_set}_model.h5'))
    
    print(f"Feature set {feature_set} completed in {time.time() - start_time:.2f} seconds")
    return feature_set, model, history, results, feature_scaler, test_dates, test_prices, df, price_scaler


def main():
    """Main function with parallel processing and optimized execution."""
    total_start_time = time.time()
    print(f"Running enhanced multi-horizon LSTM price prediction for {TICKER} using {YEARS_OF_DATA} years of data...")

    # Create output directory
    output_dir = f"{TICKER}_{YEARS_OF_DATA}yr_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data_start_time = time.time()
    df = load_data(DATA_FILE, years=YEARS_OF_DATA)
    print(f"Data loading completed in {time.time() - data_start_time:.2f} seconds")

    # Prepare arguments for parallel processing
    args_list = [(feature_set, features, df.copy(), output_dir) 
                 for feature_set, features in FEATURE_SETS.items()]

    # Initialize result containers
    models = {}
    histories = {}
    all_results = {}
    feature_scalers = {}
    test_dates_all = None
    test_prices_all = None
    price_scaler_final = None

    # Process feature sets in parallel
    parallel_start_time = time.time()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_feature_set, args_list))
    print(f"Parallel processing completed in {time.time() - parallel_start_time:.2f} seconds")

    # Collect results
    for feature_set, model, history, results_dict, feature_scaler, test_dates, test_prices, df, price_scaler in results:
        models[feature_set] = model
        histories[feature_set] = history
        all_results[feature_set] = results_dict
        feature_scalers[feature_set] = feature_scaler
        if test_dates_all is None:
            test_dates_all = test_dates
            test_prices_all = test_prices
            price_scaler_final = price_scaler

    # Generate plots and predictions
    plot_start_time = time.time()
    plot_comparison_results(PREDICTION_HORIZONS, all_results, test_dates_all, test_prices_all, output_dir=output_dir)
    print(f"Comparison plots completed in {time.time() - plot_start_time:.2f} seconds")

    prediction_start_time = time.time()
    future_predictions = predict_future_with_all_models(df, models, feature_scalers, price_scaler_final, 
                                                      PREDICTION_HORIZONS, all_results)
    print(f"Future predictions completed in {time.time() - prediction_start_time:.2f} seconds")

    plot_future_start_time = time.time()
    plot_future_comparison(df, future_predictions, PREDICTION_HORIZONS, output_dir=output_dir)
    print(f"Future prediction plots completed in {time.time() - plot_future_start_time:.2f} seconds")

    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    return future_predictions, all_results

if __name__ == "__main__":
    start_time = time.time()
    future_predictions, all_results = main()
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTotal execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")