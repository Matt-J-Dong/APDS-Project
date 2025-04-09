# Random Forest Stock Prediction
# This script uses Random Forest to predict both price levels and direction
# using only technical indicators and sentiment features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime, timedelta
import os
import math

# Configuration
TICKER = 'AAPL'
DATA_FILE = f'{TICKER}_with_sentiment.csv'
YEARS_OF_DATA = 10  # Set to either 5 or 10 years
SEQUENCE_LENGTH = 10  # Lookback window for creating features
PREDICTION_HORIZONS = [1, 5, 10]  # Days to predict into the future
RANDOM_SEED = 42
N_ESTIMATORS = 200  # Number of trees in Random Forest

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Define the feature sets to use - only technical indicators and sentiment
FEATURE_SETS = {
    'technicals_only': ['RSI', 'MACD', 'ADX', 'OBV', '+DI', '-DI', 'EMA_12', 'EMA_26'],
    'sentiment_only': ['positive', 'negative', 'neutral', 'score'],
    'technicals_and_sentiment': ['RSI', 'MACD', 'ADX', 'OBV', '+DI', '-DI', 'EMA_12', 'EMA_26',
                                'positive', 'negative', 'neutral', 'score']
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

def add_lagged_features(df, features, sequence_length=SEQUENCE_LENGTH):
    """Add lagged versions of features for the Random Forest model."""
    print("Adding lagged features...")
    df_features = df.copy()
    
    # Add lagged features
    for feature in features:
        for lag in range(1, sequence_length + 1):
            lag_name = f"{feature}_lag_{lag}"
            df_features[lag_name] = df_features[feature].shift(lag)
    
    # Drop rows with NaN from the lagged features
    df_features = df_features.dropna()
    
    return df_features

def prepare_price_data(df, features, horizons=PREDICTION_HORIZONS, sequence_length=SEQUENCE_LENGTH):
    """Prepare data for price prediction with the specified feature set."""
    print(f"Preparing price prediction data for feature set: {features}")
    
    # Ensure features exist in the dataframe
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe columns: {df.columns.tolist()}")
    
    # Add lagged features
    df_with_lags = add_lagged_features(df, features, sequence_length)
    
    # Create feature columns (original + lagged)
    feature_columns = []
    for feature in features:
        feature_columns.append(feature)
        for lag in range(1, sequence_length + 1):
            feature_columns.append(f"{feature}_lag_{lag}")
    
    # Create target columns for each horizon
    for horizon in horizons:
        df_with_lags[f'target_{horizon}d'] = df_with_lags['close_x'].shift(-horizon)
    
    # Drop rows with NaN in targets
    df_with_lags = df_with_lags.dropna()
    
    # Split data into features and targets
    X = df_with_lags[feature_columns]
    Y = {horizon: df_with_lags[f'target_{horizon}d'] for horizon in horizons}
    
    # Split into train and test (chronological split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    Y_train = {h: y.iloc[:train_size] for h, y in Y.items()}
    Y_test = {h: y.iloc[train_size:] for h, y in Y.items()}
    
    # Get dates for test set
    test_dates = df_with_lags['Date_x'].iloc[train_size:]
    
    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")
    
    return X_train, X_test, Y_train, Y_test, test_dates, df_with_lags

def prepare_direction_data(df, features, horizons=PREDICTION_HORIZONS, sequence_length=SEQUENCE_LENGTH):
    """Prepare data for direction prediction with the specified feature set."""
    print(f"Preparing direction prediction data for feature set: {features}")
    
    # Ensure features exist in the dataframe
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe columns: {df.columns.tolist()}")
    
    # Add lagged features
    df_with_lags = add_lagged_features(df, features, sequence_length)
    
    # Create feature columns (original + lagged)
    feature_columns = []
    for feature in features:
        feature_columns.append(feature)
        for lag in range(1, sequence_length + 1):
            feature_columns.append(f"{feature}_lag_{lag}")
    
    # Create direction target columns for each horizon
    for horizon in horizons:
        price_target = df_with_lags['close_x'].shift(-horizon)
        df_with_lags[f'direction_{horizon}d'] = (price_target > df_with_lags['close_x']).astype(int)
    
    # Drop rows with NaN in targets
    df_with_lags = df_with_lags.dropna()
    
    # Split data into features and targets
    X = df_with_lags[feature_columns]
    Y = {horizon: df_with_lags[f'direction_{horizon}d'] for horizon in horizons}
    
    # Split into train and test (chronological split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    Y_train = {h: y.iloc[:train_size] for h, y in Y.items()}
    Y_test = {h: y.iloc[train_size:] for h, y in Y.items()}
    
    # Get dates for test set
    test_dates = df_with_lags['Date_x'].iloc[train_size:]
    
    # Check class balance
    for horizon in horizons:
        up_ratio = Y[horizon].mean() * 100
        print(f"{horizon}-day direction: {up_ratio:.1f}% UP, {100-up_ratio:.1f}% DOWN")
    
    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")
    
    return X_train, X_test, Y_train, Y_test, test_dates, df_with_lags

def train_price_model(X_train, Y_train, horizon):
    """Train a Random Forest model for price prediction."""
    print(f"Training price model for {horizon}-day horizon...")
    
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1  # Use all cores
    )
    
    # Fit the model
    model.fit(X_train, Y_train)
    
    return model

def train_direction_model(X_train, Y_train, horizon):
    """Train a Random Forest model for direction prediction."""
    print(f"Training direction model for {horizon}-day horizon...")
    
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1  # Use all cores
    )
    
    # Fit the model
    model.fit(X_train, Y_train)
    
    return model

def evaluate_price_model(model, X_test, Y_test, horizon):
    """Evaluate the Random Forest price prediction model."""
    print(f"Evaluating price model for {horizon}-day horizon...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(Y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)
    mape = np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100
    
    # Calculate direction accuracy - fixed to ensure indices match
    # Convert predictions to a Series with the same index as Y_test
    y_pred_series = pd.Series(y_pred, index=Y_test.index)
    
    # Calculate directions
    actual_direction = (Y_test > Y_test.shift(1)).astype(int)
    pred_direction = (y_pred_series > Y_test.shift(1)).astype(int)
    
    # Drop NaN values after shift operation and align indices
    direction_data = pd.DataFrame({
        'actual': actual_direction,
        'predicted': pred_direction
    }).dropna()
    
    # Calculate accuracy only if there are valid direction predictions
    if not direction_data.empty:
        direction_accuracy = accuracy_score(
            direction_data['actual'], 
            direction_data['predicted']
        )
    else:
        direction_accuracy = np.nan
    
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'direction_accuracy': direction_accuracy,
        'predictions': y_pred,
        'actual_prices': Y_test.values
    }

def evaluate_direction_model(model, X_test, Y_test, horizon):
    """Evaluate the Random Forest direction prediction model."""
    print(f"Evaluating direction model for {horizon}-day horizon...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of UP
    
    # Calculate metrics
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, zero_division=0)
    recall = recall_score(Y_test, y_pred, zero_division=0)
    f1 = f1_score(Y_test, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(Y_test, y_pred)
    
    # Random baseline (always predict majority class)
    random_accuracy = max(Y_test.mean(), 1 - Y_test.mean())
    improvement = ((accuracy - random_accuracy) / random_accuracy) * 100
    
    print(f"Accuracy: {accuracy:.4f} ({improvement:+.1f}% vs. random)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'random_accuracy': random_accuracy,
        'improvement_vs_random': improvement,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_prob
    }

def analyze_feature_importance(models, X_train, mode='price'):
    """Analyze feature importance from Random Forest models."""
    print(f"\nAnalyzing feature importance for {mode} prediction...")
    
    # Initialize dictionary to store importances
    all_importances = {}
    
    # For each horizon and model, extract feature importances
    for horizon, model in models.items():
        # Get feature names
        feature_names = X_train.columns
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        
        # Store in dictionary
        all_importances[horizon] = {
            'features': sorted_features,
            'importances': sorted_importances
        }
        
        # Print top 10 features
        print(f"\nTop 10 features for {horizon}-day {mode} prediction:")
        for i in range(min(10, len(sorted_features))):
            print(f"  {sorted_features[i]}: {sorted_importances[i]:.4f}")
    
    return all_importances

def plot_feature_importance(importances, mode, n_features=15):
    """Plot feature importance for each horizon."""
    for horizon, data in importances.items():
        # Get top n features
        features = data['features'][:n_features]
        values = data['importances'][:n_features]
        
        # Create dataframe for plotting
        df_plot = pd.DataFrame({
            'Feature': features,
            'Importance': values
        })
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=df_plot)
        plt.title(f'Top {n_features} Features for {horizon}-Day {mode.capitalize()} Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig(f'{TICKER}_{YEARS_OF_DATA}yr_rf_{mode}_{horizon}day_importance.png')
        plt.close()

def plot_comparison_results(price_results, direction_results, test_dates):
    """Plot comparison of results across feature sets."""
    # Plot price prediction metrics
    price_metrics = ['rmse', 'mae', 'mape', 'r2']
    price_metric_names = ['RMSE ($)', 'MAE ($)', 'MAPE (%)', 'R² Score']
    
    for horizon in price_results:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(price_metrics, price_metric_names)):
            # Extract values
            values = []
            for feature_set in FEATURE_SETS:
                values.append(price_results[horizon][feature_set][metric])
            
            # Create dataframe for plotting
            df_plot = pd.DataFrame({
                'Feature Set': list(FEATURE_SETS.keys()),
                name: values
            })
            
            # Plot
            ax = sns.barplot(x='Feature Set', y=name, data=df_plot, ax=axes[i])
            axes[i].set_title(f'{name} for {horizon}-Day Price Prediction')
            
            # Format x-labels
            axes[i].set_xticklabels([x.get_text().replace('_', ' ').title() for x in axes[i].get_xticklabels()])
            
            # Set y-axis limits for R²
            if metric == 'r2':
                axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{TICKER}_{YEARS_OF_DATA}yr_rf_price_{horizon}day_metrics.png')
        plt.close()
    
    # Plot direction prediction metrics
    direction_metrics = ['accuracy', 'precision', 'recall', 'f1']
    direction_metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for horizon in direction_results:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(direction_metrics, direction_metric_names)):
            # Extract values
            values = []
            random_baseline = None
            
            for feature_set in FEATURE_SETS:
                values.append(direction_results[horizon][feature_set][metric])
                if random_baseline is None and metric == 'accuracy':
                    random_baseline = direction_results[horizon][feature_set]['random_accuracy']
            
            # Create dataframe for plotting
            df_plot = pd.DataFrame({
                'Feature Set': list(FEATURE_SETS.keys()),
                name: values
            })
            
            # Plot
            ax = sns.barplot(x='Feature Set', y=name, data=df_plot, ax=axes[i])
            axes[i].set_title(f'{name} for {horizon}-Day Direction Prediction')
            
            # Format x-labels
            axes[i].set_xticklabels([x.get_text().replace('_', ' ').title() for x in axes[i].get_xticklabels()])
            
            # Set y-axis limits
            axes[i].set_ylim(0, 1)
            
            # Add random baseline for accuracy
            if metric == 'accuracy' and random_baseline is not None:
                axes[i].axhline(y=random_baseline, color='r', linestyle='--', 
                                label=f'Random ({random_baseline:.3f})')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(f'{TICKER}_{YEARS_OF_DATA}yr_rf_direction_{horizon}day_metrics.png')
        plt.close()
    
    # Plot predictions vs actual for each feature set
    for horizon in price_results:
        plt.figure(figsize=(14, 7))
        
        # Plot actual prices
        actual_prices = list(price_results[horizon].values())[0]['actual_prices']
        plt.plot(test_dates[horizon], actual_prices, label='Actual', color='blue', linewidth=2)
        
        # Plot predictions for each feature set
        colors = ['green', 'purple', 'orange']
        linestyles = ['--', '-.', ':']
        
        for i, feature_set in enumerate(FEATURE_SETS):
            predictions = price_results[horizon][feature_set]['predictions']
            plt.plot(test_dates[horizon], predictions, 
                     label=f'{feature_set.replace("_", " ").title()}', 
                     color=colors[i % len(colors)], 
                     linestyle=linestyles[i % len(linestyles)])
        
        plt.title(f'{TICKER} {horizon}-Day Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{TICKER}_{YEARS_OF_DATA}yr_rf_price_{horizon}day_predictions.png')
        plt.close()

def predict_future(models, latest_data, horizons):
    """Predict future prices and directions using the trained models."""
    print("\nPredicting future values...")
    
    future_predictions = {'price': {}, 'direction': {}}
    
    # Current information
    current_date = latest_data['Date_x'].iloc[-1]
    current_price = latest_data['close_x'].iloc[-1]
    
    print(f"Current date: {current_date.strftime('%Y-%m-%d')}")
    print(f"Current price: ${current_price:.2f}")
    
    # Predict for each horizon and feature set
    for mode in ['price', 'direction']:
        future_predictions[mode] = {horizon: {} for horizon in horizons}
        
        for horizon in horizons:
            future_date = current_date + pd.Timedelta(days=horizon)
            print(f"\n{horizon}-Day Horizon ({future_date.strftime('%Y-%m-%d')}):")
            
            for feature_set in FEATURE_SETS:
                model = models[mode][horizon][feature_set]
                X = latest_data[feature_set]
                
                if mode == 'price':
                    predicted_price = model.predict(X)[0]
                    change = predicted_price - current_price
                    change_pct = (change / current_price) * 100
                    direction = "UP" if change > 0 else "DOWN"
                    
                    future_predictions[mode][horizon][feature_set] = {
                        'price': predicted_price,
                        'change': change,
                        'change_pct': change_pct,
                        'direction': direction
                    }
                    
                    print(f"  {feature_set.replace('_', ' ').title()}: ${predicted_price:.2f} ({change_pct:+.2f}%, {direction})")
                
                else:  # direction
                    direction_pred = model.predict(X)[0]
                    probability = model.predict_proba(X)[0][1]  # Probability of UP
                    direction = "UP" if direction_pred == 1 else "DOWN"
                    confidence = probability if direction == "UP" else 1 - probability
                    
                    future_predictions[mode][horizon][feature_set] = {
                        'direction': direction,
                        'confidence': confidence * 100,
                        'probability_up': probability * 100
                    }
                    
                    print(f"  {feature_set.replace('_', ' ').title()}: {direction} (Confidence: {confidence*100:.1f}%)")
    
    return future_predictions, current_date, current_price

def main():
    """Main function to run the Random Forest prediction pipeline."""
    print(f"Running Random Forest prediction for {TICKER} using {YEARS_OF_DATA} years of data...")
    print(f"Using only technical indicators and sentiment features as requested.")
    
    # Load data
    df = load_data(DATA_FILE, years=YEARS_OF_DATA)
    
    # Initialize dictionaries to store results
    price_models = {horizon: {} for horizon in PREDICTION_HORIZONS}
    direction_models = {horizon: {} for horizon in PREDICTION_HORIZONS}
    price_results = {horizon: {} for horizon in PREDICTION_HORIZONS}
    direction_results = {horizon: {} for horizon in PREDICTION_HORIZONS}
    test_dates = {}
    
    # Process each feature set
    for feature_set_name, features in FEATURE_SETS.items():
        print(f"\n{'='*50}")
        print(f"Processing feature set: {feature_set_name}")
        print(f"Features: {features}")
        print(f"{'='*50}\n")
        
        # Prepare data for price prediction
        X_train_price, X_test_price, Y_train_price, Y_test_price, dates_price, df_price = prepare_price_data(
            df, features, horizons=PREDICTION_HORIZONS
        )
        
        # Train and evaluate price models for each horizon
        for horizon in PREDICTION_HORIZONS:
            # Train model
            price_model = train_price_model(X_train_price, Y_train_price[horizon], horizon)
            price_models[horizon][feature_set_name] = price_model
            
            # Evaluate model
            price_result = evaluate_price_model(price_model, X_test_price, Y_test_price[horizon], horizon)
            price_results[horizon][feature_set_name] = price_result
            
            # Store test dates
            if horizon not in test_dates:
                test_dates[horizon] = dates_price
        
        # Prepare data for direction prediction
        X_train_dir, X_test_dir, Y_train_dir, Y_test_dir, dates_dir, df_dir = prepare_direction_data(
            df, features, horizons=PREDICTION_HORIZONS
        )
        
        # Train and evaluate direction models for each horizon
        for horizon in PREDICTION_HORIZONS:
            # Train model
            direction_model = train_direction_model(X_train_dir, Y_train_dir[horizon], horizon)
            direction_models[horizon][feature_set_name] = direction_model
            
            # Evaluate model
            direction_result = evaluate_direction_model(direction_model, X_test_dir, Y_test_dir[horizon], horizon)
            direction_results[horizon][feature_set_name] = direction_result
    
    # Analyze feature importance
    price_importance = {}
    direction_importance = {}
    
    # Analyze for each horizon using the 'technicals_and_sentiment' model
    for horizon in PREDICTION_HORIZONS:
        if 'technicals_and_sentiment' in price_models[horizon]:
            price_importance[horizon] = {
                'features': X_train_price.columns,
                'importances': price_models[horizon]['technicals_and_sentiment'].feature_importances_
            }
        
        if 'technicals_and_sentiment' in direction_models[horizon]:
            direction_importance[horizon] = {
                'features': X_train_dir.columns,
                'importances': direction_models[horizon]['technicals_and_sentiment'].feature_importances_
            }
    
    # Plot feature importance
    plot_feature_importance(price_importance, 'price')
    plot_feature_importance(direction_importance, 'direction')
    
    # Plot comparison results
    plot_comparison_results(price_results, direction_results, test_dates)
    
    # Prepare latest data for future prediction
    latest_data = {feature_set: add_lagged_features(df, FEATURE_SETS[feature_set])[list(X_train_price.columns)].iloc[-1:] 
                  for feature_set in FEATURE_SETS}
    
    # Predict future values
    future_predictions, current_date, current_price = predict_future(
        {'price': price_models, 'direction': direction_models},
        {'Date_x': df['Date_x'], 'close_x': df['close_x'], **latest_data},
        PREDICTION_HORIZONS
    )
    
    # Save comprehensive results to file
    with open(f'{TICKER}_{YEARS_OF_DATA}yr_rf_prediction_results.txt', 'w') as f:
        f.write(f"Random Forest Prediction Results for {TICKER}\n")
        f.write(f"Using {YEARS_OF_DATA} years of historical data\n")
        f.write(f"Using only technical indicators and sentiment features\n")
        f.write("="*70 + "\n\n")
        
        # Current price info
        f.write(f"Current Price (as of {current_date.strftime('%Y-%m-%d')}): ${current_price:.2f}\n\n")
        
        # Future predictions
        f.write("FUTURE PRICE PREDICTIONS\n")
        f.write("-"*70 + "\n\n")
        
        for horizon in PREDICTION_HORIZONS:
            future_date = current_date + pd.Timedelta(days=horizon)
            f.write(f"{horizon}-Day Horizon ({future_date.strftime('%Y-%m-%d')}):\n")
            
            for feature_set in FEATURE_SETS:
                pred = future_predictions['price'][horizon][feature_set]
                f.write(f"  {feature_set.replace('_', ' ').title()}: ${pred['price']:.2f} ({pred['change_pct']:+.2f}%, {pred['direction']})\n")
            
            f.write("\n")
        
        f.write("\nFUTURE DIRECTION PREDICTIONS\n")
        f.write("-"*70 + "\n\n")
        
        for horizon in PREDICTION_HORIZONS:
            future_date = current_date + pd.Timedelta(days=horizon)
            f.write(f"{horizon}-Day Horizon ({future_date.strftime('%Y-%m-%d')}):\n")
            
            for feature_set in FEATURE_SETS:
                pred = future_predictions['direction'][horizon][feature_set]
                f.write(f"  {feature_set.replace('_', ' ').title()}: {pred['direction']} (Confidence: {pred['confidence']:.1f}%)\n")
            
            f.write("\n")

if __name__ == "__main__":
    main()