import unittest
import pandas as pd
from scripts.amharic_labeler import AmharicNERLabeler  # Replace with your actual module name

class TestAmharicNERLabeler(unittest.TestCase):

    def setUp(self):
        """
        Initialize the AmharicNERLabeler class before each test.
        """
        self.labeler = AmharicNERLabeler()

    def test_label_tokens_price(self):
        """
        Test token labeling for price-related entities.
        """
        tokens = ['ዋጋ', '4800', 'ብር', 'ስቶቭ']
        expected_labels = [
            ('ዋጋ', 'B-PRICE'),
            ('4800', 'I-PRICE'),
            ('ብር', 'I-PRICE'),
            ('ስቶቭ', 'B-PRODUCT')
        ]
        labeled = self.labeler.label_tokens(tokens)
        self.assertEqual(labeled, expected_labels)

    def test_label_tokens_location(self):
        """
        Test token labeling for location-related entities.
        """
        tokens = ['አዲስ', 'አበባ', 'ቦሌ', 'በረራ']
        expected_labels = [
            ('አዲስ', 'B-LOCATION'),
            ('አበባ', 'O'),
            ('ቦሌ', 'B-LOCATION'),
            ('በረራ', 'B-LOCATION')
        ]
        labeled = self.labeler.label_tokens(tokens)
        self.assertEqual(labeled, expected_labels)

    def test_label_tokens_product(self):
        """
        Test token labeling for product-related entities.
        """
        tokens = ['ስቶቭ', 'ዋጋ', '3000', 'ብር']
        expected_labels = [
            ('ስቶቭ', 'B-PRODUCT'),
            ('ዋጋ', 'B-PRICE'),
            ('3000', 'I-PRICE'),
            ('ብር', 'I-PRICE')
        ]
        labeled = self.labeler.label_tokens(tokens)
        self.assertEqual(labeled, expected_labels)

    def test_label_dataframe(self):
        """
        Test labeling tokens in a pandas DataFrame.
        """
        df = pd.DataFrame({
            'tokens': [
                ['ዋጋ', '4800', 'ብር', 'ስቶቭ'],
                ['አዲስ', 'ቦሌ', 'ብር']
            ]
        })
        expected_output = [
            [('ዋጋ', 'B-PRICE'), ('4800', 'I-PRICE'), ('ብር', 'I-PRICE'), ('ስቶቭ', 'B-PRODUCT')],
            [('አዲስ', 'B-LOCATION'), ('ቦሌ', 'B-LOCATION'), ('ብር', 'I-PRICE')]
        ]

        labeled_df = self.labeler.label_dataframe(df, 'tokens')
        self.assertEqual(labeled_df['Labeled'].tolist(), expected_output)

    def test_save_conll_format(self):
        """
        Test saving the labeled data in CoNLL format.
        """
        df = pd.DataFrame({
            'Labeled': [
                [('ዋጋ', 'B-PRICE'), ('4800', 'I-PRICE'), ('ብር', 'I-PRICE')],
                [('አዲስ', 'B-LOCATION'), ('ቦሌ', 'B-LOCATION')]
            ]
        })
        file_path = 'test_output.conll'
        self.labeler.save_conll_format(df, file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            output = f.read()

        expected_output = """ዋጋ B-PRICE
4800 I-PRICE
ብር I-PRICE

አዲስ B-LOCATION
ቦሌ B-LOCATION

"""
        self.assertEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()
