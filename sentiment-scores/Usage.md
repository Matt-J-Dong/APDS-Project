# Sentiment Analysis Script - Usage

This document provides detailed instructions for using the `sentiment_1.py` script for sentiment analysis on financial headlines.

## Prerequisites

- **Python 3.x** installed.
- Required packages:
  - `torch`
  - `transformers`
  - `pandas`
  - `tqdm`
  - `matplotlib`
  - `yfinance`
  - `numpy`

Install the necessary packages using:

```bash
pip install torch transformers pandas tqdm matplotlib yfinance numpy
```

## Input Data Format

Your input CSV file must include at least the following columns:

- **Time Data**: Date and time information.
- **Headline**: Text data for sentiment analysis.

**Example:**

```csv
Time Data,Headline
2023-04-01 12:00:00,Example headline for analysis.
2023-04-01 13:00:00,Another example headline.
```

## Running the Script

The script accepts two command-line arguments:

- **Input CSV File Path**: The path to the CSV file containing your data.
- **Output CSV File Path**: The path where the output CSV file with sentiment scores will be saved.

Run the script using the following command:

```bash
python sentiment_1.py <input_csv_path> <output_csv_path>
```

**Example:**

```bash
python sentiment_1.py samsung_data_aggregated.csv samsung_sentiment_scores.csv
```

## What the Script Does

### Model Setup
Loads a pretrained BERT model, fine-tuned on financial social media comments, to classify headlines as positive, neutral, or negative.

### Data Loading
Reads the input CSV file via a custom dataset class that expects "Time Data" and "Headline" columns.

### Batch Processing
Utilizes a DataLoader to handle the data in batches, optimizing performance during sentiment prediction.

### Sentiment Prediction
Processes each headline with the sentiment classifier to obtain sentiment scores.

### Score Calculation
Computes a normalized sentiment score using the formula:

$$
\text{score} = \frac{\text{positive} - \text{negative}}{\text{positive} + \text{negative}}
$$

This score provides a balance between positive and negative sentiments.

### Output
Aggregates the sentiment scores along with their corresponding timestamps into a pandas DataFrame, which is then saved as a CSV file at the specified output path.

## Troubleshooting

### Column Name Issues
Verify that your input CSV file uses the exact column names "Time Data" and "Headline".

### Memory/Batched Processing
If you encounter memory issues during batch processing, try reducing the batch size in the DataLoader.

### GPU vs. CPU
For systems without a GPU, update the sentiment pipeline's `device` parameter to `-1` to force CPU usage.
