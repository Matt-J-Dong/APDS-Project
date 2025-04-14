# Enhanced Multi-Horizon LSTM Stock Price Prediction
# This script allows configuring the amount of historical data used (5 or 10 years)
# and compares predictions using different feature sets.
# All output files (plots, models, summaries, etc.) are saved in a subdirectory
# named after the ticker (e.g., "AAPL_outputs").

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
TICKER = 'AMD'
DATA_FILE = f'{TICKER}_with_sentiment.csv'
YEARS_OF_DATA = 10  # Set to either 5 or 10 years
SEQUENCE_LENGTH = 15  # Number of previous days to use
PREDICTION_HORIZONS = [1, 3]  # Days to predict into the future
RANDOM_SEED = 42
NUM_EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10  # Early stopping patience

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

    return df


def prepare_multi_horizon_data(df, features, horizons=PREDICTION_HORIZONS, seq_length=SEQUENCE_LENGTH):
    """Prepare data for multi-horizon prediction with the specified feature set."""
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe columns: {df.columns.tolist()}")

    feature_data = df[features].values
    close_prices = df['close_x'].values.reshape(-1, 1)

    # Scale features and target
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(feature_data)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = price_scaler.fit_transform(close_prices)

    X = []
    Y = {horizon: [] for horizon in horizons}

    for i in range(len(scaled_features) - seq_length - max(horizons)):
        X.append(scaled_features[i:i + seq_length])
        for horizon in horizons:
            Y[horizon].append(scaled_prices[i + seq_length + horizon - 1, 0])

    X = np.array(X)
    Y = {horizon: np.array(y) for horizon, y in Y.items()}

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train = {horizon: y[:train_size] for horizon, y in Y.items()}
    Y_test = {horizon: y[train_size:] for horizon, y in Y.items()}

    test_dates = {}
    test_prices = {}

    for horizon in horizons:
        test_dates[horizon] = df['Date_x'].iloc[
                              train_size + seq_length + horizon - 1:train_size + seq_length + horizon - 1 + len(X_test)]
        test_prices[horizon] = df['close_x'].iloc[
                               train_size + seq_length + horizon - 1:train_size + seq_length + horizon - 1 + len(
                                   X_test)]

    print(f"Feature set: {features}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    return X_train, X_test, Y_train, Y_test, test_dates, test_prices, price_scaler, feature_scaler, df


def build_multi_output_model(input_shape, num_outputs):
    """Build an LSTM model with multiple outputs for different horizons."""
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_outputs))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()
    return model


def train_model(model, X_train, Y_train, X_test, Y_test, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                patience=PATIENCE, output_dir="."):
    """Train the LSTM model with early stopping and save the loss curve to the output directory."""
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

    # Plot training & validation loss and save to output_dir
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

    return model, history


def evaluate_model_on_training_data(model, X_train, Y_train, price_scaler, horizons, num_predictors):
    """Evaluate training performance for each horizon, including R² and adjusted R²."""
    y_train_pred_scaled = model.predict(X_train)
    results = {}
    for i, horizon in enumerate(sorted(horizons)):
        horizon_pred_scaled = y_train_pred_scaled[:, i].reshape(-1, 1)
        horizon_pred_reshaped = np.zeros((len(horizon_pred_scaled), 1))
        horizon_pred_reshaped[:, 0] = horizon_pred_scaled.flatten()
        horizon_pred = price_scaler.inverse_transform(horizon_pred_reshaped).flatten()
        actual_train = price_scaler.inverse_transform(Y_train[horizon].reshape(-1, 1)).flatten()
        mse = mean_squared_error(actual_train, horizon_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(actual_train, horizon_pred)
        r2 = r2_score(actual_train, horizon_pred)
        mape = np.mean(np.abs((actual_train - horizon_pred) / actual_train)) * 100
        n = len(actual_train)
        p = num_predictors
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


def evaluate_model(model, X_test, Y_test, price_scaler, test_prices, horizons, num_predictors):
    """Evaluate the model's performance on the test set for each horizon, including adjusted R²."""
    y_pred_scaled = model.predict(X_test)
    results = {}
    for i, horizon in enumerate(sorted(horizons)):
        horizon_pred_scaled = y_pred_scaled[:, i].reshape(-1, 1)
        horizon_pred_reshaped = np.zeros((len(horizon_pred_scaled), 1))
        horizon_pred_reshaped[:, 0] = horizon_pred_scaled.flatten()
        horizon_pred = price_scaler.inverse_transform(horizon_pred_reshaped).flatten()
        actual_prices = test_prices[horizon].values
        mse = mean_squared_error(actual_prices, horizon_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(actual_prices, horizon_pred)
        r2 = r2_score(actual_prices, horizon_pred)
        mape = np.mean(np.abs((actual_prices - horizon_pred) / actual_prices)) * 100
        n = len(actual_prices)
        p = num_predictors
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


def plot_comparison_results(horizons, all_results, test_dates, test_prices, output_dir="."):
    """Plot performance comparison between feature sets and save plots to output_dir."""
    metrics = ['rmse', 'mae', 'mape', 'r2']
    metric_names = ['RMSE ($)', 'MAE ($)', 'MAPE (%)', 'R² Score']
    for horizon in sorted(horizons):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            metric_values = []
            for feature_set in FEATURE_SETS.keys():
                metric_values.append(all_results[feature_set][horizon][metric])
            df_plot = pd.DataFrame({
                'Feature Set': list(FEATURE_SETS.keys()),
                name: metric_values
            })
            sns.barplot(x='Feature Set', y=name, data=df_plot, ax=axes[i])
            axes[i].set_title(f'{name} for {horizon}-Day Horizon')
            axes[i].grid(True, axis='y')
            if metric != 'r2':
                baseline = metric_values[0]
                for j, value in enumerate(metric_values):
                    if j > 0:
                        improvement = ((baseline - value) / baseline) * 100
                        axes[i].text(j, value, f"{improvement:.1f}%", ha='center', va='bottom')
            else:
                baseline = metric_values[0]
                for j, value in enumerate(metric_values):
                    if j > 0:
                        improvement = ((value - baseline) / abs(baseline)) * 100
                        axes[i].text(j, value, f"+{improvement:.1f}%", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_{horizon}day_metric_comparison.png'))
        plt.close()

    for horizon in sorted(horizons):
        plt.figure(figsize=(16, 8))
        plt.plot(test_dates[horizon], test_prices[horizon], label='Actual Price', color='blue', linewidth=2)
        colors = ['red', 'green', 'purple', 'orange']
        linestyles = ['--', '-.', ':', '-']
        for i, feature_set in enumerate(FEATURE_SETS.keys()):
            plt.plot(test_dates[horizon], all_results[feature_set][horizon]['predictions'],
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


def predict_future_with_all_models(df, models, feature_scalers, price_scaler, horizons):
    """Predict future prices using all models and return a dictionary of predictions."""
    print("\nPredicting future prices with all models...")
    future_predictions = {}
    for feature_set, model in models.items():
        features = FEATURE_SETS[feature_set]
        feature_scaler = feature_scalers[feature_set]
        last_sequence = df[features].values[-SEQUENCE_LENGTH:, :]
        last_sequence_scaled = feature_scaler.transform(last_sequence)
        X_pred = last_sequence_scaled.reshape(1, SEQUENCE_LENGTH, len(features))
        predictions_scaled = model.predict(X_pred)[0]
        future_predictions[feature_set] = {}
        for i, horizon in enumerate(sorted(horizons)):
            horizon_pred_scaled = predictions_scaled[i].reshape(1, 1)
            horizon_pred_reshaped = np.zeros((1, 1))
            horizon_pred_reshaped[0, 0] = horizon_pred_scaled[0, 0]
            predicted_price = price_scaler.inverse_transform(horizon_pred_reshaped)[0, 0]
            current_price = df['close_x'].iloc[-1]
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            direction = "UP" if change > 0 else "DOWN"
            last_date = df['Date_x'].iloc[-1]
            future_date = last_date + pd.Timedelta(days=horizon)
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


def plot_future_comparison(df, future_predictions, horizons, output_dir="."):
    """Plot future price predictions from all models and save the plot to output_dir."""
    historical_dates = df['Date_x'].iloc[-90:]  # Last 90 days
    historical_prices = df['close_x'].iloc[-90:]
    plt.figure(figsize=(16, 8))
    plt.plot(historical_dates, historical_prices, label='Historical Price', color='blue', linewidth=2)
    colors = ['red', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'd']
    last_date = df['Date_x'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=h) for h in sorted(horizons)]
    for i, feature_set in enumerate(FEATURE_SETS.keys()):
        dates = [last_date]
        prices = [df['close_x'].iloc[-1]]
        for horizon in sorted(horizons):
            pred = future_predictions[feature_set][horizon]
            dates.append(pred['prediction_date'])
            prices.append(pred['predicted_price'])
        plt.plot(dates, prices, label=f'{feature_set.replace("_", " ").title()} Prediction',
                 color=colors[i], linestyle='--', marker=markers[i])
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
    plt.savefig(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_future_predictions.png'))
    plt.close()


def main():
    """Main function to run the enhanced multi-horizon LSTM prediction pipeline."""
    print(f"Running enhanced multi-horizon LSTM price prediction for {TICKER} using {YEARS_OF_DATA} years of data...")

    # Create a subdirectory named after the ticker for all outputs
    output_dir = f"{TICKER}_outputs"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(DATA_FILE, years=YEARS_OF_DATA)

    models = {}
    histories = {}
    all_results = {}
    feature_scalers = {}

    for feature_set, features in FEATURE_SETS.items():
        print(f"\n{'=' * 50}")
        print(f"Processing feature set: {feature_set}")
        print(f"Features: {features}")
        print(f"{'=' * 50}\n")

        X_train, X_test, Y_train, Y_test, test_dates, test_prices, price_scaler, feature_scaler, df = prepare_multi_horizon_data(
            df, features, horizons=PREDICTION_HORIZONS, seq_length=SEQUENCE_LENGTH
        )

        feature_scalers[feature_set] = feature_scaler

        model = build_multi_output_model((SEQUENCE_LENGTH, len(features)), len(PREDICTION_HORIZONS))

        model, history = train_model(model, X_train, Y_train, X_test, Y_test, patience=PATIENCE, output_dir=output_dir)

        results = evaluate_model(model, X_test, Y_test, price_scaler, test_prices, PREDICTION_HORIZONS,
                                 num_predictors=len(features))

        models[feature_set] = model
        histories[feature_set] = history
        all_results[feature_set] = results

        train_results = evaluate_model_on_training_data(model, X_train, Y_train, price_scaler, PREDICTION_HORIZONS,
                                                        num_predictors=len(features))
        print(f"\nTraining performance for {feature_set}:")
        for horizon in sorted(PREDICTION_HORIZONS):
            print(f"{horizon}-Day Horizon:")
            print(f"  Train RMSE: ${train_results[horizon]['rmse']:.2f}")
            print(f"  Train MAE: ${train_results[horizon]['mae']:.2f}")
            print(f"  Train MAPE: {train_results[horizon]['mape']:.2f}%")
            print(f"  Train R²: {train_results[horizon]['r2']:.4f}")
            print(f"  Train Adjusted R²: {train_results[horizon]['adjusted_r2']:.4f}")

        print(f"\nTesting performance for {feature_set}:")
        for horizon in sorted(PREDICTION_HORIZONS):
            print(f"{horizon}-Day Horizon:")
            print(f"  RMSE: ${results[horizon]['rmse']:.2f}")
            print(f"  MAE: ${results[horizon]['mae']:.2f}")
            print(f"  MAPE: {results[horizon]['mape']:.2f}%")
            print(f"  R²: {results[horizon]['r2']:.4f}")
            print(f"  Adjusted R²: {results[horizon]['adjusted_r2']:.4f}")

        model.save(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_{feature_set}_model.h5'))

    plot_comparison_results(PREDICTION_HORIZONS, all_results, test_dates, test_prices, output_dir=output_dir)
    future_predictions = predict_future_with_all_models(df, models, feature_scalers, price_scaler, PREDICTION_HORIZONS)
    plot_future_comparison(df, future_predictions, PREDICTION_HORIZONS, output_dir=output_dir)

    with open(os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_enhanced_predictions.txt'), 'w') as f:
        f.write(f"Enhanced Multi-Horizon LSTM Stock Price Prediction for {TICKER}\n")
        f.write(f"Using {YEARS_OF_DATA} years of historical data\n")
        f.write("=" * 70 + "\n\n")

        current_price = df['close_x'].iloc[-1]
        current_date = df['Date_x'].iloc[-1]
        f.write(f"Current Price (as of {current_date.strftime('%Y-%m-%d')}): ${current_price:.2f}\n\n")

        f.write("FEATURE SET COMPARISON\n")
        f.write("-" * 70 + "\n\n")
        for horizon in sorted(PREDICTION_HORIZONS):
            f.write(f"Horizon: {horizon} Days\n")
            f.write("-" * 50 + "\n")
            for metric in ['rmse', 'mae', 'mape', 'r2']:
                metric_name = {'rmse': 'RMSE ($)', 'mae': 'MAE ($)', 'mape': 'MAPE (%)', 'r2': 'R² Score'}[metric]
                baseline = all_results['price_only'][horizon][metric]
                f.write(f"{metric_name}:\n")
                for feature_set in FEATURE_SETS.keys():
                    value = all_results[feature_set][horizon][metric]
                    if feature_set == 'price_only':
                        f.write(f"  {feature_set.replace('_', ' ').title()}: {value:.4f}\n")
                    else:
                        if metric != 'r2':
                            improvement = ((baseline - value) / baseline) * 100
                            f.write(
                                f"  {feature_set.replace('_', ' ').title()}: {value:.4f} ({improvement:.2f}% improvement)\n")
                        else:
                            improvement = ((value - baseline) / abs(baseline)) * 100
                            f.write(
                                f"  {feature_set.replace('_', ' ').title()}: {value:.4f} ({improvement:.2f}% improvement)\n")
                f.write("\n")
            f.write("\n")

        f.write("FUTURE PREDICTIONS\n")
        f.write("-" * 70 + "\n\n")
        for horizon in sorted(PREDICTION_HORIZONS):
            f.write(
                f"{horizon}-Day Horizon ({future_predictions['price_only'][horizon]['prediction_date'].strftime('%Y-%m-%d')}):\n")
            for feature_set in FEATURE_SETS.keys():
                pred = future_predictions[feature_set][horizon]
                f.write(
                    f"  {feature_set.replace('_', ' ').title()}: ${pred['predicted_price']:.2f} ({pred['change_pct']:.2f}%, {pred['direction']})\n")
            f.write("\n")

        f.write("DATASET INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Data range: {df['Date_x'].min().strftime('%Y-%m-%d')} to {df['Date_x'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"Total data points: {len(df)}\n")
        f.write(f"Training data: {int(len(df) * 0.8)} points ({80}%)\n")
        f.write(f"Testing data: {len(df) - int(len(df) * 0.8)} points ({20}%)\n")

    print(
        f"\nEnhanced prediction complete! Results saved to {os.path.join(output_dir, f'{TICKER}_{YEARS_OF_DATA}yr_enhanced_predictions.txt')}")

    return future_predictions, all_results


if __name__ == "__main__":
    main()
