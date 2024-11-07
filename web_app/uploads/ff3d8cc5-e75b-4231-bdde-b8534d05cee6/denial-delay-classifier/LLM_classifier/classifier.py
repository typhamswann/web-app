from api import APIClient
from prompts import PromptTuner
import pandas as pd

class Classifier:
    def __init__(self):
        self.client = APIClient()
        self.prompt_tuner = PromptTuner()
        
    def classify(self, prompt, few_shot=True):
        response = ""
        if not few_shot:
            response = self.client.get_response(self.prompt_tuner.zero_shot_instructions(), prompt)
        else:
            response = self.client.get_response(self.prompt_tuner.get_few_shot_instructions(), prompt)
        return response
        
classifier = Classifier()

# Classify congress Validate set
validation_set = pd.read_csv("data_labeled/think_tank/validation_original.csv")

#clean
validation_set["label"] = validation_set["label"].apply(lambda x: int(x))
validation_set["text"] = validation_set["text"].apply(lambda x: x.replace("\n"," "))

#Classify a sample.
validation_set = validation_set.sample(20)

validation_set_labeled = pd.DataFrame(columns=["text", "human_label" "llm_label"])

for counter, (_, row) in enumerate(validation_set.iterrows()):
    try:
        text = row["text"]
        human_label = row["label"]
        llm_label = int(classifier.classify(text,few_shot=True))
    
    except Exception as e:
        print("ERROR: ", e)
        llm_label = 999
    
    new_row_df = pd.DataFrame([{"text": text, "human_label": human_label, "llm_label": llm_label}])
    validation_set_labeled = pd.concat([validation_set_labeled, new_row_df], ignore_index=True)
        
    if counter % 5== 0:
        print("Classified", counter, "/", len(validation_set), "paragraphs")

validation_set_labeled.to_csv("LLM_classifier/LLM_results/original_validation.csv", mode='a', index=False)

    