import sys
import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer, pipeline
import time

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(
    "StephanAkkerman/FinTwitBERT-sentiment",
    num_labels=3,
    id2label={0: "neutral", 1: "positive", 2: "negative"},
    label2id={"neutral": 0, "positive": 1, "negative": 2},
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("StephanAkkerman/FinTwitBERT-sentiment")

sentiment_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,        # GPU (if available)
    top_k=None,
    padding=True,
    truncation=True,
    max_length=256
)

# Dataset
class MyTextDataset(Dataset):
    def __init__(self, csv_file_path):
        self.data = []
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append((row["Headline"], row["Time Data"]))
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# Main execution
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python sentiment_opt.py <input_csv_path> <output_csv_path>")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]

    dataset = MyTextDataset(input_csv_path)
    loader = DataLoader(dataset, batch_size=32)

    log_positive_score = []
    log_neutral_score = []
    log_negative_score = []
    log_datetime = []

    start_time = time.time()

    for headlines, times in tqdm(loader):

        # Fast inference with AMP
        with torch.cuda.amp.autocast():
            outputs = sentiment_classifier(list(headlines), batch_size=len(headlines))

        for output, td in zip(outputs, times):
            for label in output:
                if label["label"] == 'positive':
                    log_positive_score.append(label["score"])
                elif label["label"] == 'neutral':
                    log_neutral_score.append(label["score"])
                elif label["label"] == 'negative':
                    log_negative_score.append(label["score"])
            log_datetime.append(td)

    end_time = time.time()
    print(f"\n\nExecution Time: {end_time - start_time:.2f} seconds\n\n")

    df = pd.DataFrame({
        'Datetime': log_datetime,
        'positive': log_positive_score,
        'neutral': log_neutral_score,
        'negative': log_negative_score,
    })

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    df['score'] = (df['positive'] - df['negative']) / (df['positive'] + df['negative'])
    df.to_csv(output_csv_path, index=True)
