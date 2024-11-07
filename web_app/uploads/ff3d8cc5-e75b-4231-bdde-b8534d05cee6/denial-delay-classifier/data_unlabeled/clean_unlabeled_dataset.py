"""
Purpose: Cleaning up the unlabeled congress paragraphs. Some of the rows in this dataset had been 
labeled since the creation of the document, so we are removing those.

Returns: Final, unlabeled congress dataset.
"""

import pandas as pd

class UnlabeledcDataCleaner:
    def __init__(self, unlabeled_file, labeled_file, output_file):
        """
        Initialize the CongressDataCleaner with file paths.

        :param unlabeled_file: Path to the CSV file containing unlabeled data.
        :param labeled_file: Path to the CSV file containing labeled data.
        :param output_file: Path to the CSV file where cleaned data will be saved.
        """
        self.unlabeled_file = unlabeled_file
        self.labeled_file = labeled_file
        self.output_file = output_file

    def clean(self):
        """Clean the unlabeled data by removing rows present in the labeled dataset and save the result."""
        # Load data
        df_unlabeled = pd.read_csv(self.unlabeled_file)
        df_labeled = pd.read_csv(self.labeled_file)

        # Find common paragraph IDs
        common_paragraph_ids = df_unlabeled[df_unlabeled['paragraph_id'].isin(df_labeled['id'])]

        # Remove the rows with common paragraph IDs from the unlabeled data
        df_unlabeled = df_unlabeled[~df_unlabeled['paragraph_id'].isin(common_paragraph_ids['paragraph_id'])].reset_index(drop=True)

        # Save the cleaned data
        df_unlabeled.to_csv(self.output_file, index=False)

if __name__ == "__main__":
    # Define file paths
    unlabeled_file = "data_unlabeled/final_paragraphs_meta.csv"
    labeled_file = "data_labeled/brown_congress_labels_mapped.csv"
    output_file = "data_unlabeled/unlabeled_congress.csv"

    # Create a CongressDataCleaner instance and clean the data
    cleaner = UnlabeledcDataCleaner(unlabeled_file, labeled_file, output_file)
    cleaner.clean()
