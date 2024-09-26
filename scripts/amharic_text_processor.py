import re
import nltk
import numpy as np  # Import numpy for NaN
import pandas as pd
from nltk.tokenize import word_tokenize

# Download NLTK resources if you haven't already
nltk.download('punkt')

class AmharicTextPreprocessor:
    """
    A class for preprocessing Amharic text, including normalization, tokenization, metadata extraction, 
    and structured storage.
    
    Attributes:
    -----------
    punctuation : re.Pattern
        A regex pattern to match Amharic-specific punctuation and symbols to be removed.
    amharic_pattern : re.Pattern
        A regex pattern to match Amharic Unicode characters for filtering non-Amharic text.
    """
    
    def __init__(self):
        """
        Initializes the AmharicTextPreprocessor with predefined punctuation and Amharic character patterns.
        """
        # Define regex for common Amharic punctuations and symbols that need to be removed
        self.punctuation = re.compile(r'[፡።፣፤፥፦፧፨,፠፨፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፿]')
        
        # Regex pattern to match Amharic Unicode block (U+1200 to U+137F)
        self.amharic_pattern = re.compile(r'[\u1200-\u137F]+')

    def normalize_text(self, text):
        """
        Normalizes Amharic text by removing unwanted punctuation, normalizing spaces, 
        and filtering only Amharic words.
        
        Parameters:
        -----------
        text : str
            The input text to normalize.
        
        Returns:
        --------
        str
            The normalized Amharic text with only Amharic words.
        """
        if not isinstance(text, str):  # Skip non-string values
            return np.nan

        # Remove unwanted punctuation and symbols
        text = re.sub(self.punctuation, '', text)
        
        # Remove extra spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract only Amharic words using Amharic Unicode pattern
        amharic_words = self.amharic_pattern.findall(text)
        
        return ' '.join(amharic_words)

    def tokenize_text(self, text):
        """
        Tokenizes Amharic text into individual words using NLTK's word_tokenize.
        
        Parameters:
        -----------
        text : str
            The normalized Amharic text to tokenize.
        
        Returns:
        --------
        list
            A list of tokenized words.
        """
        if not isinstance(text, str):  # Skip non-string values
            return np.nan
        
        # Tokenize the normalized text
        tokens = word_tokenize(text)
        
        return tokens

    def preprocess(self, text):
        """
        Preprocesses the input text by normalizing and tokenizing it.
        
        Parameters:
        -----------
        text : str
            The raw input text to be preprocessed.
        
        Returns:
        --------
        list or np.nan
            A list of tokenized words from the normalized Amharic text or NaN for invalid inputs.
        """
        # First normalize the text
        normalized_text = self.normalize_text(text)
        
        # If normalization returns NaN, return NaN directly
        if pd.isna(normalized_text):
            return np.nan
        
        # Then tokenize the normalized text
        tokens = self.tokenize_text(normalized_text)
        
        return tokens

    def extract_metadata(self, raw_text):
        """
        Extracts metadata such as sender and timestamp from raw text.
        
        Assumes the input follows a format like:
        'Sender: [name], Timestamp: [time], Message: [message]'
        
        Parameters:
        -----------
        raw_text : str
            The raw text containing metadata and message.
        
        Returns:
        --------
        dict
            A dictionary containing the extracted metadata (sender, timestamp) and message content.
        """
        if not isinstance(raw_text, str):  # Skip non-string values
            return {'sender': np.nan, 'timestamp': np.nan, 'message': np.nan}

        # Example pattern for extracting metadata: "Sender: <name>, Timestamp: <time>, Message: <message>"
        pattern = r"Sender:\s*(?P<sender>.*?),\s*Timestamp:\s*(?P<timestamp>.*?),\s*Message:\s*(?P<message>.*)"
        match = re.match(pattern, raw_text)
        
        if match:
            return match.groupdict()  # Returns a dict: {'sender': ..., 'timestamp': ..., 'message': ...}
        else:
            return {'sender': np.nan, 'timestamp': np.nan, 'message': raw_text}

    def preprocess_dataframe(self, df, text_column):
        """
        Applies the preprocessing pipeline (normalization and tokenization) to a DataFrame column.
        Separates metadata (sender, timestamp) from the message content.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the text data to be preprocessed.
        text_column : str
            The column name in the DataFrame where Amharic text is stored.
        
        Returns:
        --------
        pandas.DataFrame
            A DataFrame with additional columns for preprocessed (tokenized) text, sender, and timestamp.
        """
        def preprocess_row(row):
            # Extract metadata (sender, timestamp, message)
            metadata = self.extract_metadata(row)
            
            # Preprocess the message content
            metadata['preprocessed_message'] = self.preprocess(metadata['message'])
            
            return metadata
        
        # Apply the preprocessing pipeline to each row in the specified text column
        metadata_df = pd.DataFrame(df[text_column].apply(preprocess_row).tolist())
        
        # Merge the original DataFrame with the metadata columns
        df = pd.concat([df, metadata_df], axis=1)
        
        return df
