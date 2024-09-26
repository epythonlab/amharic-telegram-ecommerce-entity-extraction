import pandas as pd
import numpy as np
import re

class AmharicNERLabeler:
    def __init__(self):
        # You can define custom patterns for matching products, prices, and locations here
        self.price_pattern = re.compile(r'\d+[\s]*ብር', 'ዋጋ')  # Simplified pattern for price detection
        self.location_list = ['አዲስ', 'ቦሌ', 'ቡልጋሪ', 'በረራ']  # Add more locations if necessary
        self.product_keywords = ['ምርት', 'ምርቶች']  # Add more product-related keywords
        
    
    def label_tokens(self, tokens):
        """
        Label each token in the tokenized text based on NER criteria (Product, Price, Location).
        """
        labels = []
        for token in tokens:
            # Price labeling
            if re.match(self.price_pattern, token):
                labels.append("B-PRICE")
            # Product labeling (simplified, can expand)
            elif token in self.product_keywords:
                labels.append("B-Product")
            # Location labeling
            elif token in self.location_list:
                labels.append("B-LOC")
            else:
                labels.append("O")  # Non-entity words

        return list(zip(tokens, labels))

    def label_dataframe(self, df, token_column):
        """
        Label tokens for each message in a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the tokenized text.
        token_column : str
            The column name in the DataFrame where tokenized Amharic text is stored.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with tokens and their corresponding NER labels.
        """
        df['Labeled'] = df[token_column].apply(self.label_tokens)
        return df
    
    def save_conll_format(self, labeled_data, file_path):
        """
        Save the labeled data in CoNLL format.

        Parameters:
        -----------
        labeled_data : pandas.DataFrame
            The DataFrame containing token-label pairs.
        file_path : str
            The path to the file where the CoNLL format will be saved.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in labeled_data.iterrows():
                for token, label in row['Labeled']:
                    f.write(f"{token} {label}\n")
                f.write("\n")  # Blank line between sentences/messages
