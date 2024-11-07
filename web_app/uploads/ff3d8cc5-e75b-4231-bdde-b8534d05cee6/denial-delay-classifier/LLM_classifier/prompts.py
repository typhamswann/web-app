class PromptTuner():
    def __init__(self):
        self.zero_shot_inststructions = [ 
        {"role": "system", "content":"""Categories: ```Denial (2): This includes any paragraphs that implies that global warming is not happening, that human greenhouse gas emissions don't cause global warming, or that the impacts of global warming are not bad.

Delay (1): This includes paragraphs implying any kind of opposition to climate efforts that are not denial. Includes attacks on climate solutions, policies and technologies as well as arguments that center on the benefits and need for fossil fuels. It also includes attacks on the climate movement and science (science or movement is unreliable or not settled or calling it conspiracy.)
        
No claim (0): this includes paragraphs that contain information that enables climate efforts, neutral stances, or paragraphs not related to climate.```

Edge Cases: ```
- Paragraphs characterizing the movement as unreliable (such as saying media, politicians or environmentalist are biased or alarmist) belongs to delay.
- Paragraphs emphasizing the doubt, unreliability or bias of climate science and scientists belongs to denial. Examples: Climate Science is unsettled. Climate scientists are biased. There is no climate consesus. It's a conspiracy and hoax.```

You are a climate change information labeler. You are labeling a dataset of potential climate misinformation paragraphs. The categories are denial, delay, or no misinformation, as described above. You will label denial with "2", delay with "1", and no misinformation with "0." Remember to consider the edge cases. Respond with only the number. It is very important your response is only the number of the category identified.
         """},
       ]
        
        self.few_shot_instructions = [
              {"role": "system", "content":"""Categories: ```Denial (2): This includes any paragraphs that implies that global warming is not happening, that human greenhouse gas emissions don't cause global warming, or that the impacts of global warming are not bad.

Delay (1): This includes paragraphs implying any kind of opposition to climate efforts that are not denial. Includes attacks on climate solutions, policies and technologies as well as arguments that center on the benefits and need for fossil fuels. It also includes attacks on the climate movement and science (science or movement is unreliable or not settled or calling it conspiracy.)
        
No claim (0): this includes paragraphs that contain information that enables climate efforts, neutral stances, or paragraphs not related to climate.```

Edge Cases: ```
- Paragraphs characterizing the movement as unreliable (such as saying media, politicians or environmentalist are biased or alarmist) belongs to delay.
- Paragraphs emphasizing the doubt, unreliability or bias of climate science and scientists belongs to denial. Examples: Climate Science is unsettled. Climate scientists are biased. There is no climate consesus. It's a conspiracy and hoax.```

You are a climate change information labeler. You are labeling a dataset of potential climate misinformation paragraphs. The categories are denial, delay, or no misinformation, as described above. You will label denial with "2", delay with "1", and no misinformation with "0." Remember to consider the edge cases. Respond with only the number. It is very important your response is only the number of the category identified."""},
              
            {
                "role": "user",
                "content": "Though the sunspots are affecting the radiation received from the sun, they are not the only thing affecting our climate. Much closer to home, there are thousands of volcanos and volcanic vents spewing volcanic ash and gases into the atmosphere."
            }, 
            
            {
                "role": "assistant",
                "content": "2"
            },
            
            {
                "role": "user",
                "content": "We have a great solution to America's current energy crisis, and we  do have one. And the great news is that the answer is here in our  backyard. We have more coal in the United States than any other country  in the world. We have abundant sources of natural gas"
            }, 
            
            {
                "role": "assistant",
                "content": "1"
            },
            
            {
                "role": "user",
                "content": "The Russian cutoff of natural gas to Europe was also a reminder of the geopolitical risk of discouraging domestic coal under cap and trade."
            }, 
            
            {
                "role": "assistant",
                "content": "1"
            }
        ]
    
    def get_zero_shot_instructions(self):
        return self.zero_shot_inststructions
    
    def get_few_shot_instructions(self):
        return self.few_shot_instructions