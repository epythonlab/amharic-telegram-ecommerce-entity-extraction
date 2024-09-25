import re
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources if you haven't already
nltk.download('punkt')

class AmharicTextPreprocessor:
    def __init__(self):
        # Define common Amharic punctuations, symbols, etc., that need to be removed or replaced
        self.punctuation = re.compile(r'[፡።፣፤፥፦፧፨,፠፨፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፿]')

    def normalize_text(self, text):
        """
        Normalizes the Amharic text by removing unwanted characters and normalizing spaces.
        """
        # Remove punctuation and special Amharic characters
        text = re.sub(self.punctuation, '', text)
        
        # Normalize spaces (remove extra spaces and new lines)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize_text(self, text):
        """
        Tokenizes Amharic text using word tokenization.
        """
        tokens = word_tokenize(text)
        return tokens

    def preprocess(self, text):
        """
        Main function that normalizes and tokenizes Amharic text.
        """
        normalized_text = self.normalize_text(text)
        tokens = self.tokenize_text(normalized_text)
        
        return tokens
