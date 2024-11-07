import pandas as pd
import ast
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np


class DataPrepper:
    def __init__(self):
        pass
    
    def map_label(self, label:str):
        "Maps the CARDS taxonomy to the denial (2), delay(1) no claim (0) categories"
        
        if label.startswith("1") or label.startswith("2") or label.startswith("3"):
            return 2
        elif label.startswith("4"):
            return 1
        elif label.startswith("5"):
            if label == "5_3_2" or label == "5_2_5" or label.startswith("5_1"):
                return 2
            else:
                return 1
        else: #Category 0. No claim.
            return 0
            
    def mode_count(self, row):
        labels = [row['coder_0'], row['coder_1'], row['coder_2']]
        
        # Count occurrences of each label
        label_counts = pd.Series(labels).value_counts()
        
        # Find the maximum occurrence
        max_count = label_counts.max()
        
        # Find all labels with the maximum occurrence
        common_labels = label_counts[label_counts == max_count].index.tolist()
        
        # Return the mode count and the label(s) if the count is at least 2
        if max_count >= 2:
            return max_count, common_labels[0]  # Assuming you want the first if there are multiple modes
        return max_count, None
    
    def calculate_intercoder_reliability(self, df, coder_columns):
        """
        Calculates Fleiss' Kappa for given columns in a dataframe.

        Args:
        df (pd.DataFrame): The dataframe containing the data.
        coder_columns (list): List of column names that contain coder's ratings.

        Returns:
        float: The Fleiss' Kappa score indicating intercoder reliability.
        """
        # Create a new DataFrame to count occurrences of each category for each row
        labels_list = []
        
        # Iterate over each row and count occurrences of each category
        for _, row in df[coder_columns].iterrows():
            # Append the value_counts of each row to the list
            labels_list.append(row.value_counts())
        
        # Create a DataFrame from the list and replace NaNs with 0s
        labels_df = pd.concat(labels_list, axis=1).fillna(0).T

        # Calculate Fleiss' Kappa
        kappa = fleiss_kappa(labels_df.values, method='fleiss')

        return kappa

    
data_prepper = DataPrepper()

congress_df = pd.read_csv("data_labeled/brown_congress_labels_mapped.csv")

#Transform each label of each coder to denial/delay/nothing
congress_df["coder_0"] = congress_df["coder_0"].apply(lambda x: ast.literal_eval(x))
congress_df["coder_0"] = congress_df["coder_0"].apply(lambda x: [data_prepper.map_label(label) for label in x])
congress_df["coder_0"] = congress_df["coder_0"].apply(lambda x: list(set(x)))

#Keep one label. single-label classification . Priority ranking: denial > delay > nothing.
congress_df['coder_0'] = congress_df['coder_0'].apply(lambda x: max(x))


congress_df["coder_1"] = congress_df["coder_1"].apply(lambda x: ast.literal_eval(x))
congress_df["coder_1"] = congress_df["coder_1"].apply(lambda x: [data_prepper.map_label(label) for label in x])
congress_df["coder_1"] = congress_df["coder_1"].apply(lambda x: list(set(x))) #Keep unique values.

congress_df['coder_1'] = congress_df['coder_1'].apply(lambda x: max(x))


congress_df["coder_2"] = congress_df["coder_2"].apply(lambda x: ast.literal_eval(x))
congress_df["coder_2"] = congress_df["coder_2"].apply(lambda x: [data_prepper.map_label(label) for label in x])
congress_df["coder_2"] = congress_df["coder_2"].apply(lambda x: list(set(x)))

congress_df['coder_2'] = congress_df['coder_2'].apply(lambda x: max(x))


#Add column with agreed upon categories.
congress_df['agreement_level_simple'], congress_df['label_simple'] = zip(*congress_df.apply(data_prepper.mode_count, axis=1))

#Calculate intercoder reliability
kappa_score = data_prepper.calculate_intercoder_reliability(congress_df, ["coder_0", "coder_1", "coder_2"])
print("Fleiss' Kappa:", kappa_score)

#Total congress paragraphs
print("Paragraphs labeled:", congress_df.shape[0])

#Prepare congress for validating and testing
congress_clean = congress_df[congress_df["agreement_level_simple"] > 1] # Remove paragraphs with no agreement.
congress_clean = congress_clean[["text", "label_simple"]].rename(columns={"label_simple":"label"})

#Split in half, one for validating and one for testing.
congress_clean = congress_clean.sample(frac=1, random_state=42).reset_index(drop=True) #shuffle.
split_index = len(congress_clean) // 2  # Find the midpoint of the DataFrame

# Create validation and test sets
validation_set = congress_clean[:split_index].reset_index()
test_set = congress_clean[split_index:].reset_index()

validation_set.to_csv("data_labeled/congress/validation.csv")
test_set.to_csv("data_labeled/congress/test.csv")


#Prepare think tank dataset
tt_df = pd.read_csv("data_labeled/CARDS2_multisource_multilabel_data_final.csv")
tt_df = tt_df[tt_df["corpus"]=="cards_single"]

#Transform each label to denial/delay/nothing
tt_df["labels_final"] = tt_df["labels_final"].apply(lambda x: ast.literal_eval(x))
tt_df["labels_final"] = tt_df["labels_final"].apply(lambda x: [data_prepper.map_label(label) for label in x][0])
tt_df = tt_df[["text", "labels_final"]].rename(columns={"labels_final":"label"})

# Create validation and test sets 
tt_df = tt_df.sample(frac=1, random_state=42).reset_index(drop=True) #shuffle.
split_index = len(tt_df) // 2  # Find the midpoint of the DataFrame

# 
validation_set = tt_df[:split_index].reset_index()
test_set = tt_df[split_index:].reset_index()

validation_set.to_csv("data_labeled/think_tank/validation.csv")
test_set.to_csv("data_labeled/think_tank/test.csv")

#Prepare originals cards validate
tt_original_validation = pd.read_csv("data_labeled/original_cards/validation.csv")
#Transform each label to denial/delay/nothing
tt_original_validation["label"] = tt_original_validation["claim"].apply(lambda x: data_prepper.map_label(x))
tt_original_validation = tt_original_validation[["text", "label"]]

tt_original_validation.to_csv("data_labeled/think_tank/validation_original.csv")


