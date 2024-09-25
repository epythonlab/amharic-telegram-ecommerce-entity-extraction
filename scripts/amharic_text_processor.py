import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

# Download NLTK resources if you haven't already
nltk.download('punkt')

class AmharicTextPreprocessor:
    def __init__(self):
        # Define common Amharic punctuations, symbols, etc., that need to be removed or replaced
        self.punctuation = re.compile(r'[፡።፣፤፥፦፧፨,፠፨፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፿]')
        self.amharic_pattern = re.compile(r'[\u1200-\u137F]+')  # Regex for Amharic Unicode block

    def normalize_text(self, text):
        """
        Normalizes the Amharic text by removing unwanted characters and normalizing spaces.
        """
        if not isinstance(text, str):  # Handle non-string values
            return text
        # Remove punctuation and special Amharic characters
        text = re.sub(self.punctuation, '', text)
        
        # Normalize spaces (remove extra spaces and new lines)
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract only Amharic words
        amharic_words = self.amharic_pattern.findall(text)
        return ' '.join(amharic_words)

    def tokenize_text(self, text):
        """
        Tokenizes Amharic text using word tokenization.
        """
        if not isinstance(text, str):  # Handle non-string values
            return text
        tokens = word_tokenize(text)
        return tokens

    def preprocess(self, text):
        """
        Main function that normalizes and tokenizes Amharic text.
        """
        normalized_text = self.normalize_text(text)
        tokens = self.tokenize_text(normalized_text)
        
        return tokens

    def preprocess_dataframe(self, df, text_column):
        """
        Preprocess all texts in a specified column of a pandas DataFrame.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the text data.
        text_column : str
            The column name in the DataFrame where Amharic text is stored.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with an additional column for preprocessed (tokenized) text.
        """
        # Apply the preprocessing to each row in the specified text column
        df[f'{text_column}_preprocessed'] = df[text_column].apply(self.preprocess)
        return df

