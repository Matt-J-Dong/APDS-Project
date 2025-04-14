import sys
import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer, pipeline, BertTokenizer

# =============================================================================
# Set up the sentiment analysis model
# =============================================================================

# Load a pretrained BERT model fine-tuned for sentiment analysis on financial social media data.
model = BertForSequenceClassification.from_pretrained(
    "StephanAkkerman/FinTwitBERT-sentiment",
    num_labels=3,
    id2label={0: "neutral", 1: "positive", 2: "negative"},
    label2id={"neutral": 0, "positive": 1, "negative": 2},
)
model.eval()  # Set the model to evaluation mode

# Load the corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained("StephanAkkerman/FinTwitBERT-sentiment")

# Construct a Huggingface pipeline for text classification with the sentiment model.
# Using device=0 enables GPU acceleration (if available).
sentiment_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,          # Use GPU if available, else set device=-1 for CPU
    top_k=None,        # Return all classification labels with scores
    padding=True,      # Pad sequences to the maximum length in the batch
    truncation=True,   # Truncate sequences longer than max_length
    max_length=256     # Maximum sequence length
)

# =============================================================================
# Custom Dataset Definition
# =============================================================================

class MyTextDataset(Dataset):
    """
    Custom Dataset class to load text and corresponding time data from a CSV file.
    The CSV file should contain at least two columns: "Time Data" and "Headline".
    """
    def __init__(self, csv_file_path):
        self.data = []

        # Open and read the CSV file using DictReader for key-value access
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract time data and headline from each row
                time_data = row["Time Data"]
                headline = row["Headline"]
                # Append the tuple (headline, time_data) to the data list
                self.data.append((headline, time_data))

        # Store the total number of samples
        self.num_samples = len(self.data)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieve the sample at the given index.
        Returns:
            tuple: (headline, time_data)
        """
        headline, time_data = self.data[idx]
        return headline, time_data

# =============================================================================
# Main Execution: Command-Line Arguments & Processing
# =============================================================================

if __name__ == '__main__':
    # Check if the correct number of command-line arguments are provided.
    # sys.argv[0] is the script name, sys.argv[1] is the input CSV file,
    # and sys.argv[2] is the output CSV file.
    if len(sys.argv) != 3:
        print("Usage: python sentiment_1.py <input_csv_path> <output_csv_path>")
        sys.exit(1)

    # Get the file paths from command-line arguments
    input_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]

    # Create an instance of the custom dataset and a DataLoader for batching
    dataset = MyTextDataset(input_csv_path)
    loader = DataLoader(dataset, batch_size=32)

    # Initialize lists to log scores and corresponding datetime values
    log_positive_score = []
    log_neutral_score = []
    log_negative_score = []
    log_datetime = []

    # Iterate over batches of data from the DataLoader
    for (input_data, time_data) in tqdm(loader):
        # Use automatic mixed precision for faster computation on CUDA-enabled GPUs
        with torch.cuda.amp.autocast():
            # Run the sentiment classifier on the batch of input headlines
            outputs = sentiment_classifier(list(input_data), batch_size=len(input_data))

        # Process each output and its corresponding time data
        for (output, td) in zip(outputs, time_data):
            # Each 'output' contains a list of label-score dictionaries
            for label in output:
                if label["label"] == 'positive':
                    log_positive_score.append(label["score"])
                elif label["label"] == 'neutral':
                    log_neutral_score.append(label["score"])
                elif label["label"] == 'negative':
                    log_negative_score.append(label["score"])

            # Log the corresponding datetime for the batch
            log_datetime.append(td)

    # =============================================================================
    # Data Aggregation and Saving
    # =============================================================================

    # Create a DataFrame to aggregate datetime and sentiment scores
    df = pd.DataFrame({
        'Datetime': log_datetime,
        'positive': log_positive_score,
        'neutral': log_neutral_score,
        'negative': log_negative_score,
    })

    # Convert the 'Datetime' column to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Set 'Datetime' as the DataFrame index and sort the DataFrame by datetime
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    # Compute a sentiment score as the normalized difference between positive and negative scores
    df['score'] = (df['positive'] - df['negative']) / (df['positive'] + df['negative'])

    # Save the sentiment scores DataFrame to the specified CSV file.
    df.to_csv(output_csv_path, index=True)
