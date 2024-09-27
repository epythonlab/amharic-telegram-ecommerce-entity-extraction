import unittest
import pandas as pd
import numpy as np
from scripts.amharic_text_processor import AmharicTextPreprocessor  # Assuming this is in amharic_preprocessor.py

class TestAmharicTextPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = AmharicTextPreprocessor()

    def test_normalize_text_valid(self):
        # Test a valid Amharic string with unwanted characters
        text = "ሰላም! Hello ዋጋ 300 ብር."
        expected_result = "ሰላም ዋጋ 300 ብር"
        result = self.preprocessor.normalize_text(text)
        self.assertEqual(result, expected_result)

    def test_normalize_text_only_amharic(self):
        # Test Amharic text with valid characters
        text = "ዋጋ 300 ብር"
        expected_result = "ዋጋ 300 ብር"
        result = self.preprocessor.normalize_text(text)
        self.assertEqual(result, expected_result)

    def test_normalize_text_with_spaces(self):
        # Test Amharic text with extra spaces
        text = "   ዋጋ    300 ብር   "
        expected_result = "ዋጋ 300 ብር"
        result = self.preprocessor.normalize_text(text)
        self.assertEqual(result, expected_result)

    def test_normalize_text_with_invalid_chars(self):
        # Test text with English letters and special characters
        text = "ሰላም! @# Hello ዋጋ 300 ብር."
        expected_result = "ሰላም ዋጋ 300 ብር"
        result = self.preprocessor.normalize_text(text)
        self.assertEqual(result, expected_result)

    def test_normalize_text_empty_string(self):
        # Test empty string case
        text = ""
        result = self.preprocessor.normalize_text(text)
        self.assertTrue(np.isnan(result))

    def test_normalize_text_none_input(self):
        # Test None input case
        text = None
        result = self.preprocessor.normalize_text(text)
        self.assertTrue(np.isnan(result))

    def test_preprocess_dataframe(self):
        # Test processing a DataFrame with Amharic text
        data = {'message': ["ሰላም! ዋጋ 300 ብር.", "በአንድ ዋጋ", ""]}  # One valid, one partial, one empty
        df = pd.DataFrame(data)
        processed_df = self.preprocessor.preprocess_dataframe(df, 'message')

        expected_output = pd.Series(["ሰላም ዋጋ 300 ብር", "በአንድ ዋጋ", np.nan], name="preprocessed_message")
        pd.testing.assert_series_equal(processed_df['preprocessed_message'], expected_output)

if __name__ == '__main__':
    unittest.main()
