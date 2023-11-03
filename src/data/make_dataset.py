import torch
import pandas as pd
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# Function for preprocessing a pair of text strings
def preprocess_text_pair(row, length_threshold=50):
    # Extracting data from the row
    reference, translation = row['reference'], row['translation']
    
    # Converting 'length_diff' and 'similarity' to tensors
    length_diff = torch.tensor([row['lenght_diff']])
    similarity = torch.tensor([row['similarity']])
    
    # Tokenization of reference and translation text
    reference_tokens = word_tokenize(reference)
    translation_tokens = word_tokenize(translation)

    # Length normalization with padding '<pad>' tokens
    pad_token = '<pad>'
    padded_ref = reference_tokens[:length_threshold] + [pad_token] * max(0, length_threshold - len(reference_tokens))
    padded_trans = translation_tokens[:length_threshold] + [pad_token] * max(0, length_threshold - len(translation_tokens))

    return padded_ref, padded_trans, length_diff, similarity

# Function to preprocess the entire dataset
def preprocess_dataset(dataset, length_threshold=50):
    preprocessed_data = pd.DataFrame()

    # Apply 'preprocess_text_pair' to each row in the dataset
    (
        preprocessed_data['reference'],
        preprocessed_data['translation'],
        preprocessed_data['lenght_diff'],
        preprocessed_data['similarity']
    ) = zip(*dataset.apply(lambda row: preprocess_text_pair(row, length_threshold), axis=1))

    return preprocessed_data

# Custom Dataset Class
class ParaphraseDataset(Dataset):
    def __init__(self, data, test=False):
        self.data = data
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        reference = self.data.iloc[idx]['reference']
        translation = self.data.iloc[idx]['translation']
        length_diff = self.data.iloc[idx]['lenght_diff']
        similarity = self.data.iloc[idx]['similarity']
    
        if self.test:
                    # During testing, return only what's needed for prediction
            return {'reference': reference, 'translation': translation}
        else:
            # During training, return additional features
            return {'reference': reference, 'translation': translation, 'lenght_diff': length_diff, 'similarity': similarity}


# Main function for data preprocessing and dataset creation
def main(raw_df, train_size=100000, val_size=5000):
    preprocessed_df = preprocess_dataset(raw_df)
    
    # Split the preprocessed dataset into training and validation sets
    train_df, val_df = train_test_split(preprocessed_df, test_size=0.2, random_state=42)
    
    # Create ParaphraseDataset instances for training and validation
    train_dataset = ParaphraseDataset(train_df[:train_size])
    val_dataset = ParaphraseDataset(val_df[:val_size], True)

    return train_dataset, val_dataset
