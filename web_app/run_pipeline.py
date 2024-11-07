import os
import json
from openai import OpenAI
import re
import copy
from pathlib import PurePosixPath
from nbformat import v4 as nbf
import shutil
import nbformat
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from the environment
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)


# Takes in: directory (string)
# Returns: python_files_content (obj), file_tree_structure (obj)
def handle_files(directory):
    # Dictionary to store relative filepath as key and file content as value
    python_files_content = {}
    
    # Function to construct the filetree in JSON format
    def construct_file_tree(directory, root):
        file_tree = {
            "name": os.path.basename(directory),
            "is_directory": True,
            "children": []
        }
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Recursively construct file tree for directories
                file_tree["children"].append(construct_file_tree(item_path, root))
            else:
                # Append file information to the file tree
                file_tree["children"].append({
                    "name": item,
                    "is_directory": False,
                    "children": []
                })
                # If it's a Python file, read its content
                if item.endswith('.py'):
                    relative_path = os.path.relpath(item_path, root)
                    with open(item_path, 'r') as file:
                        python_files_content[relative_path] = file.read()
        return file_tree

    # Construct the file tree and ensure the relative paths are from the root
    file_tree_structure = construct_file_tree(directory, directory)

    return python_files_content, file_tree_structure


# Takes in: python_files_content (obj)
# Returns: python_files_content (obj)
def comment_code(python_files_content):
    """
    Loops through the dictionary, applies the `comment_code` function to each 
    Python code content string, and returns a new dictionary with the modified content.
    
    :param python_files_content: Dictionary with file paths as keys and Python code as values
    :return: New dictionary with the commented Python code
    """
    # Dictionary to store the updated content
    updated_python_files_content = {}

    # Iterate through the dictionary
    for file_path, file_content in python_files_content.items():
        # Apply the comment_code function to the file content
        updated_content = ai_comment(file_content)
        # Store the updated content in the new dictionary
        updated_python_files_content[file_path] = updated_content

    return updated_python_files_content

def ai_comment(python_file_string):
    response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": "You are a code commenter. You will see python code. You should reply with a commented version of the code. Please only include one single comment per function. It is important that you don't change the code. It is important that you respond with the entirety of the code. ",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "def generate_squares(n):\n    return [x**2 for x in range(1, n + 1)]\n\ndef filter_even_squares(squares):\n    return [square for square in squares if square % 2 == 0]\n\ndef main():\n    n = 20\n    squares = generate_squares(n)\n    even_squares = filter_even_squares(squares)\n    print(f\"Squares from 1 to {n}: {squares}\")\n    print(f\"Even squares: {even_squares}\")\n\nmain()"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "# Function to generate a list of squares for numbers from 1 to n\ndef generate_squares(n):\n    return [x**2 for x in range(1, n + 1)]\n\n# Function to filter out only even squares from a list of squares\ndef filter_even_squares(squares):\n    return [square for square in squares if square % 2 == 0]\n\n# Main function to run the program\ndef main():\n    n = 20\n    squares = generate_squares(n) \n    even_squares = filter_even_squares(squares)\n    print(f\"Squares from 1 to {n}: {squares}\")\n    print(f\"Even squares: {even_squares}\")\n\nmain()"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "import random\n\ndef generate_random_numbers(n, lower, upper):\n    return [random.randint(lower, upper) for _ in range(n)]\n\ndef calculate_mean(numbers):\n    return sum(numbers) / len(numbers)\n\ndef calculate_median(numbers):\n    sorted_numbers = sorted(numbers)\n    mid = len(sorted_numbers) // 2\n    return (sorted_numbers[mid] if len(sorted_numbers) % 2 != 0 \n            else (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2)\n\ndef calculate_variance(numbers, mean):\n    return sum((x - mean) ** 2 for x in numbers) / len(numbers)\n\ndef calculate_standard_deviation(variance):\n    return variance ** 0.5\n\ndef sort_numbers(numbers, descending=False):\n    return sorted(numbers, reverse=descending)\n\ndef generate_statistics(numbers):\n    mean = calculate_mean(numbers)\n    median = calculate_median(numbers)\n    variance = calculate_variance(numbers, mean)\n    std_deviation = calculate_standard_deviation(variance)\n    return {\n        'mean': mean,\n        'median': median,\n        'variance': variance,\n        'std_deviation': std_deviation\n    }\n\ndef print_statistics(stats):\n    for key, value in stats.items():\n        print(f\"{key.capitalize()}: {value:.2f}\")\n\ndef main():\n    n = 50\n    lower_bound = 10\n    upper_bound = 100\n    random_numbers = generate_random_numbers(n, lower_bound, upper_bound)\n    sorted_numbers = sort_numbers(random_numbers)\n    stats = generate_statistics(sorted_numbers)\n    print_statistics(stats)\n\nmain()"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "import random\n\n# Generates a list of n random integers between lower and upper bounds\ndef generate_random_numbers(n, lower, upper):\n    return [random.randint(lower, upper) for _ in range(n)]\n\n# Calculates the mean of a list of numbers\ndef calculate_mean(numbers):\n    return sum(numbers) / len(numbers)\n\n# Calculates the median of a list of numbers\ndef calculate_median(numbers):\n    sorted_numbers = sorted(numbers)\n    mid = len(sorted_numbers) // 2\n    return (sorted_numbers[mid] if len(sorted_numbers) % 2 != 0 \n            else (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2)\n\n# Calculates the variance of a list of numbers given the mean\ndef calculate_variance(numbers, mean):\n    return sum((x - mean) ** 2 for x in numbers) / len(numbers)\n\n# Calculates the standard deviation given the variance\ndef calculate_standard_deviation(variance):\n    return variance ** 0.5\n\n# Sorts a list of numbers in ascending or descending order\ndef sort_numbers(numbers, descending=False):\n    return sorted(numbers, reverse=descending)\n\n# Generates a dictionary of statistical measures for a list of numbers\ndef generate_statistics(numbers):\n    mean = calculate_mean(numbers)\n    median = calculate_median(numbers)\n    variance = calculate_variance(numbers, mean)\n    std_deviation = calculate_standard_deviation(variance)\n    return {\n        'mean': mean,\n        'median': median,\n        'variance': variance,\n        'std_deviation': std_deviation\n    }\n\n# Prints the statistical measures in a formatted manner\ndef print_statistics(stats):\n    for key, value in stats.items():\n        print(f\"{key.capitalize()}: {value:.2f}\")\n\n# Main function to generate random numbers and display their statistics\ndef main():\n    n = 50\n    lower_bound = 10\n    upper_bound = 100\n    random_numbers = generate_random_numbers(n, lower_bound, upper_bound)\n    sorted_numbers = sort_numbers(random_numbers)\n    stats = generate_statistics(sorted_numbers)\n    print_statistics(stats)\n\nmain()"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\nclass SimpleNN(nn.Module):\n    def __init__(self, input_size, hidden_size, output_size):\n        super(SimpleNN, self).__init__()\n        self.fc1 = nn.Linear(input_size, hidden_size)\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(hidden_size, output_size)\n\n    def forward(self, x):\n        x = self.fc1(x)\n        x = self.relu(x)\n        x = self.fc2(x)\n        return x\n\ninput_size = 10\nhidden_size = 20\noutput_size = 1\nlearning_rate = 0.001\nepochs = 100\n\nmodel = SimpleNN(input_size, hidden_size, output_size)\ncriterion = nn.MSELoss()\noptimizer = optim.Adam(model.parameters(), lr=learning_rate)\n\nx_train = torch.randn(100, input_size)\ny_train = torch.randn(100, output_size)\n\ndataset = TensorDataset(x_train, y_train)\ndataloader = DataLoader(dataset, batch_size=10, shuffle=True)\n\nfor epoch in range(epochs):\n    for inputs, targets in dataloader:\n        outputs = model(inputs)\n        loss = criterion(outputs, targets)\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\n# Simple Neural Network definition with one hidden layer\nclass SimpleNN(nn.Module):\n    def __init__(self, input_size, hidden_size, output_size):\n        super(SimpleNN, self).__init__()\n        self.fc1 = nn.Linear(input_size, hidden_size)\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(hidden_size, output_size)\n\n    def forward(self, x):\n        x = self.fc1(x)\n        x = self.relu(x)\n        x = self.fc2(x)\n        return x\n\ninput_size = 10\nhidden_size = 20\noutput_size = 1\nlearning_rate = 0.001\nepochs = 100\n\nmodel = SimpleNN(input_size, hidden_size, output_size)\ncriterion = nn.MSELoss()\noptimizer = optim.Adam(model.parameters(), lr=learning_rate)\n\nx_train = torch.randn(100, input_size)\ny_train = torch.randn(100, output_size)\n\ndataset = TensorDataset(x_train, y_train)\ndataloader = DataLoader(dataset, batch_size=10, shuffle=True)\n\nfor epoch in range(epochs):\n    for inputs, targets in dataloader:\n        outputs = model(inputs)\n        loss = criterion(outputs, targets)\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": python_file_string
            }
        ]
        }
    ],
    temperature=0,
    max_tokens=16380,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "text"
    }
    )

    text = response.choices[0].message.content

    if text.startswith('```python\n') and text.endswith('```'):
        # Remove the wrapping backticks and the language identifier
        return text[10:-3].strip()
    
    return text


# Takes in: python_files_content (obj)
# Returns: python_files_content (obj)
def summarize_functions(code_dict):
    # Create a new dictionary to store the original code with function information added
    updated_code_dict = {}

    for file_path, code in code_dict.items():
        # Get the function summaries for the current file using the ai_summarize_functions function
        try:
            function_info = ai_summarize_functions(code)  # Summarize the functions
        except Exception as e:
            function_info = {"error": str(e)}  # Handle any errors gracefully

        # Add the function information to the original code
        updated_code_dict[file_path] = {
            "code": code,
            "function_information": function_info["function_information"]
        }

    return updated_code_dict

def ai_summarize_functions(python_file_string):
    response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": "You are a python code summarizer. You will see python code. You should summarize all functions based on what they do with respect to the whole script.",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "import random\n\ndef generate_random_numbers(n, lower, upper):\n    return [random.randint(lower, upper) for _ in range(n)]\n\ndef calculate_mean(numbers):\n    return sum(numbers) / len(numbers)\n\ndef calculate_median(numbers):\n    sorted_numbers = sorted(numbers)\n    mid = len(sorted_numbers) // 2\n    return (sorted_numbers[mid] if len(sorted_numbers) % 2 != 0 \n            else (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2)\n\ndef calculate_variance(numbers, mean):\n    return sum((x - mean) ** 2 for x in numbers) / len(numbers)\n\ndef calculate_standard_deviation(variance):\n    return variance ** 0.5\n\ndef sort_numbers(numbers, descending=False):\n    return sorted(numbers, reverse=descending)\n\ndef generate_statistics(numbers):\n    mean = calculate_mean(numbers)\n    median = calculate_median(numbers)\n    variance = calculate_variance(numbers, mean)\n    std_deviation = calculate_standard_deviation(variance)\n    return {\n        'mean': mean,\n        'median': median,\n        'variance': variance,\n        'std_deviation': std_deviation\n    }\n\ndef print_statistics(stats):\n    for key, value in stats.items():\n        print(f\"{key.capitalize()}: {value:.2f}\")\n\ndef main():\n    n = 50\n    lower_bound = 10\n    upper_bound = 100\n    random_numbers = generate_random_numbers(n, lower_bound, upper_bound)\n    sorted_numbers = sort_numbers(random_numbers)\n    stats = generate_statistics(sorted_numbers)\n    print_statistics(stats)\n\nmain()"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "{\"function_information\":[{\"function_name\":\"generate_random_numbers\",\"function_description\":\"Generates a list of 'n' random integers between 'lower' and 'upper' bounds.\"},{\"function_name\":\"calculate_mean\",\"function_description\":\"Calculates and returns the mean (average) of a list of numbers.\"},{\"function_name\":\"calculate_median\",\"function_description\":\"Calculates and returns the median of a list of numbers.\"},{\"function_name\":\"calculate_variance\",\"function_description\":\"Calculates and returns the variance of a list of numbers based on the given mean.\"},{\"function_name\":\"calculate_standard_deviation\",\"function_description\":\"Calculates and returns the standard deviation from the variance.\"},{\"function_name\":\"sort_numbers\",\"function_description\":\"Sorts the list of numbers in ascending or descending order based on the 'descending' parameter.\"},{\"function_name\":\"generate_statistics\",\"function_description\":\"Generates and returns a dictionary containing the mean, median, variance, and standard deviation of a list of numbers.\"},{\"function_name\":\"print_statistics\",\"function_description\":\"Prints the statistical metrics (mean, median, variance, standard deviation) in a formatted manner.\"},{\"function_name\":\"main\",\"function_description\":\"Main function that ties together the generation of random numbers, sorting, calculating statistics, and printing the results.\"}]}"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": python_file_string
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "function_descriptions",
        "schema": {
            "type": "object",
            "required": [
            "function_information"
            ],
            "properties": {
            "function_information": {
                "type": "array",
                "items": {
                "type": "object",
                "required": [
                    "function_name",
                    "function_description"
                ],
                "properties": {
                    "function_name": {
                    "type": "string"
                    },
                    "function_description": {
                    "type": "string"
                    }
                },
                "additionalProperties": False
                }
            }
            },
            "additionalProperties": False
        },
        "strict": True
        }
    }
    )
    text = response.choices[0].message.content
    obj = json.loads(text)

    return obj


# Takes in: python_files_content (obj)
# Returns: python_files_content (obj)
def summarize_files(python_content_obj):
    summarized_content = {}

    for filepath, file_info in python_content_obj.items():
        # Extract relevant information for the summary
        code = file_info.get("code", "")
        function_information = file_info.get("function_information", [])

        # Generate the file summary using the ai_summarize_files function
        file_summary = ai_summarize_files(
            filetree_obj=python_content_obj,  # Assuming this is passed as the full python content
            project_function_information=function_information,
            python_text=code,
            python_filepath=filepath
        )

        # Copy existing file info and add the new summary
        summarized_content[filepath] = {
            "code": code,
            "function_information": function_information,
            "file_summary": file_summary
        }

    return summarized_content

def ai_summarize_files(filetree_obj, project_function_information, python_text, python_filepath):
    response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": "You will receive a filetree in JSON format, a list of function descriptions for the whole project in JSON form, and a python file. Please create a short summary of the what the python file does. Include all instances of outside files or functions being referenced or necessary in the python file. Please respond with only the file information.",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Filetree JSON: ```{\n  \"name\": \"DataAnalysisProject\",\n  \"children\": [\n    {\n      \"name\": \"data\",\n      \"children\": [\n        {\n          \"name\": \"dataset.csv\",\n          \"children\": [],\n          \"is_directory\": false\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"scripts\",\n      \"children\": [\n        {\n          \"name\": \"data_processing.py\",\n          \"children\": [],\n          \"is_directory\": false\n        },\n        {\n          \"name\": \"visualization.py\",\n          \"children\": [],\n          \"is_directory\": false\n        },\n        {\n          \"name\": \"model_training.py\",\n          \"children\": [],\n          \"is_directory\": false\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"output\",\n      \"children\": [\n        {\n          \"name\": \"results.txt\",\n          \"children\": [],\n          \"is_directory\": false\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"README.md\",\n      \"children\": [],\n      \"is_directory\": false\n    }\n  ],\n  \"is_directory\": true\n}```\n\nProject Function information: ```{\n    \"scripts/model_training.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"train_model\",\n                \"function_description\": \"Trains a linear regression model using input features 'X' and target values 'y'. Returns the trained model.\"\n            },\n            {\n                \"function_name\": \"predict\",\n                \"function_description\": \"Generates predictions for new input data 'X_new' using the provided trained model.\"\n            },\n            {\n                \"function_name\": \"evaluate_model\",\n                \"function_description\": \"Evaluates the performance of the trained model on test data 'X_test' and 'y_test', returning the model's score.\"\n            }\n        ]\n    },\n    \"scripts/visualization.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"plot_data\",\n                \"function_description\": \"Plots a graph of the provided data, using 'date' for the x-axis and 'value' for the y-axis, and displays the plot.\"\n            },\n            {\n                \"function_name\": \"save_plot\",\n                \"function_description\": \"Plots a graph of the provided data, using 'date' for the x-axis and 'value' for the y-axis, and saves the plot to the specified file path.\"\n            }\n        ]\n    },\n    \"scripts/data_processing.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"load_data\",\n                \"function_description\": \"Loads data from a CSV file specified by the 'filepath' and returns it as a pandas DataFrame.\"\n            },\n            {\n                \"function_name\": \"clean_data\",\n                \"function_description\": \"Cleans the data by removing rows with missing values and filtering out rows where the 'value' column is negative.\"\n            },\n            {\n                \"function_name\": \"save_clean_data\",\n                \"function_description\": \"Saves the cleaned DataFrame to a specified CSV file without including the index column.\"\n            }\n        ]\n    }\n}```\n\nscripts/model_training.py: ```from sklearn.linear_model import LinearRegression\n\ndef train_model(X, y):\n    model = LinearRegression()\n    model.fit(X, y)\n    return model\n\ndef predict(model, X_new):\n    predictions = model.predict(X_new)\n    return predictions\n\ndef evaluate_model(model, X_test, y_test):\n    score = model.score(X_test, y_test)\n    return score\n``` "
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "scripts/model_training.py is responsible for training, predicting, and evaluating a linear regression model, forming the core of the project's machine learning component. It interacts with processed data (likely prepared by data_processing.py) and its output can be visualized using functions in visualization.py. It expects cleaned and structured data as input, indicating its role after data preprocessing and before results visualization in the overall project workflow."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Filetree JSON: ```{\n  \"name\": \"UrbanDevelopmentStudy\",\n  \"children\": [\n    {\n      \"name\": \"data\",\n      \"children\": [\n        {\n          \"name\": \"population_data\",\n          \"children\": [\n            {\n              \"name\": \"city_population.csv\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"urban_growth.csv\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        },\n        {\n          \"name\": \"economic_data\",\n          \"children\": [\n            {\n              \"name\": \"gdp_per_capita.csv\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"employment_rate.csv\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"scripts\",\n      \"children\": [\n        {\n          \"name\": \"data_preprocessing\",\n          \"children\": [\n            {\n              \"name\": \"load_data.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"clean_data.py\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        },\n        {\n          \"name\": \"analysis\",\n          \"children\": [\n            {\n              \"name\": \"population_trends.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"economic_analysis.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"urban_growth_analysis.py\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        },\n        {\n          \"name\": \"visualization\",\n          \"children\": [\n            {\n              \"name\": \"plot_population.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"plot_economic.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"plot_urban_growth.py\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"results\",\n      \"children\": [\n        {\n          \"name\": \"summary_statistics\",\n          \"children\": [\n            {\n              \"name\": \"population_summary.txt\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"economic_summary.txt\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        },\n        {\n          \"name\": \"final_report.pdf\",\n          \"children\": [],\n          \"is_directory\": false\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"README.md\",\n      \"children\": [],\n      \"is_directory\": false\n    }\n  ],\n  \"is_directory\": true\n}```\n\nProject Function Information: ```{\n  \"scripts/data_preprocessing/load_data.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"load_population_data\",\n        \"function_description\": \"Loads population data from CSV files and returns them as a dictionary of pandas DataFrames.\"\n      },\n      {\n        \"function_name\": \"load_economic_data\",\n        \"function_description\": \"Loads economic data from CSV files and returns them as a dictionary of pandas DataFrames.\"\n      }\n    ]\n  },\n  \"scripts/data_preprocessing/clean_data.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"remove_missing_values\",\n        \"function_description\": \"Removes rows with missing values from all DataFrames in the input dictionary and returns the cleaned dictionary.\"\n      },\n      {\n        \"function_name\": \"filter_data_by_year\",\n        \"function_description\": \"Filters all DataFrames in the input dictionary to include data from a specified range of years and returns the filtered dictionary.\"\n      }\n    ]\n  },\n  \"scripts/analysis/population_trends.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"calculate_population_growth_rate\",\n        \"function_description\": \"Calculates the annual growth rate of the population for each city and returns the resulting DataFrame.\"\n      },\n      {\n        \"function_name\": \"identify_rapid_growth_cities\",\n        \"function_description\": \"Identifies cities with a population growth rate above a specified threshold and returns a list of these cities.\"\n      }\n    ]\n  },\n  \"scripts/analysis/economic_analysis.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"calculate_gdp_growth\",\n        \"function_description\": \"Calculates the annual GDP growth rate for each city and returns the resulting DataFrame.\"\n      },\n      {\n        \"function_name\": \"calculate_employment_rate_change\",\n        \"function_description\": \"Calculates the change in employment rate over time for each city and returns the resulting DataFrame.\"\n      }\n    ]\n  },\n  \"scripts/analysis/urban_growth_analysis.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"calculate_urban_growth_rate\",\n        \"function_description\": \"Calculates the annual urban growth rate for each city and returns the resulting DataFrame.\"\n      },\n      {\n        \"function_name\": \"correlate_population_and_urban_growth\",\n        \"function_description\": \"Calculates the correlation between population growth and urban area expansion for each city and returns a correlation matrix.\"\n      }\n    ]\n  },\n  \"scripts/visualization/plot_population.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"plot_population_trend\",\n        \"function_description\": \"Plots the population trend over time for a given city and displays the plot.\"\n      },\n      {\n        \"function_name\": \"plot_population_comparison\",\n        \"function_description\": \"Plots a comparison of population growth rates between multiple cities and displays the plot.\"\n      }\n    ]\n  },\n  \"scripts/visualization/plot_economic.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"plot_gdp_trend\",\n        \"function_description\": \"Plots the GDP trend over time for a given city and displays the plot.\"\n      },\n      {\n        \"function_name\": \"plot_employment_rate\",\n        \"function_description\": \"Plots the employment rate change over time for a given city and displays the plot.\"\n      }\n    ]\n  },\n  \"scripts/visualization/plot_urban_growth.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"plot_urban_expansion\",\n        \"function_description\": \"Plots the urban expansion trend over time for a given city and displays the plot.\"\n      },\n      {\n        \"function_name\": \"plot_population_vs_urban_growth\",\n        \"function_description\": \"Plots the relationship between population growth and urban area expansion for a given city and displays the plot.\"\n      }\n    ]\n  }\n}```\n\ndata_preprocessing/clean_data.py: ```def remove_missing_values(data_dict):\n    cleaned_data = {key: df.dropna() for key, df in data_dict.items()}\n    return cleaned_data\n\ndef filter_data_by_year(data_dict, start_year, end_year):\n    filtered_data = {key: df[(df['year'] >= start_year) & (df['year'] <= end_year)] for key, df in data_dict.items()}\n    return filtered_data\n```"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "The `clean_data.py` script is used for preprocessing datasets related to urban development, specifically focusing on handling missing data and time-specific data filtering. It contains two main functions: remove_missing_values(data_dict)` which takes a dictionary of pandas DataFrames (likely provided by `load_data.py`), iterates over each DataFrame, removes rows with any missing values, and returns a cleaned dictionary of DataFrames without missing data.\n`filter_data_by_year(data_dict, start_year, end_year)` filters the data in each DataFrame within the input dictionary to include only the rows with 'year' values between the specified start and end years, returning the filtered dictionary of DataFrames.\n\nThe script ensures that the data is clean and relevant to the specified time period before it's further analyzed or visualized in other parts of the project."
            }
        ]
        },
                {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Filetree JSON: ```{filetree_obj}```\n\nProject Function information: ```{project_function_information}```\n\n{python_filepath}: ```{python_text}``` "
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "text"
    }
    )
    text = response.choices[0].message.content
    return text


# Takes in: python_files_content (obj)
# Returns: python_files_content (obj)
def identify_references(code_dict):
    # Create a new dictionary to store the original code with function information added
    updated_code_dict = {}

    for file_path, code in code_dict.items():
        # Get the function summaries for the current file using the ai_summarize_functions function
        try:
            referenced_filepaths = ai_identify_references(code["code"])  # Summarize the functions
        except Exception as e:
            referenced_filepaths = {"error": str(e)}  # Handle any errors gracefully

        # Add the function information to the original code
        updated_code_dict[file_path] = {
            "code": code["code"],
            "function_information": code['function_information'],
            "file_summary": code['file_summary'],
            "referenced_filepaths": referenced_filepaths
        }

    updated_code_dict = helper_filter_generic_imports(updated_code_dict)

    return updated_code_dict

def ai_identify_references(python_text):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are tasked with identifying filepaths within a python file. You will see python code from a python file. You must respond with all external filepaths. It's important that you included all external filepaths identified. Be sure to include external and internal files, using the exact path used in the code. Include only filepaths themselves. Ignore variables or parameters that are not clear instances of a specific filepath mention."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "import pandas as pd\nimport matplotlib.pyplot as plt\nfrom helpers.helper import preprocess_data, calculate_summary_statistics\n\n# Load external data (CSV file)\ndef load_data(file_path):\n    \"\"\"\n    Load CSV data into a pandas DataFrame.\n    \n    Parameters:\n        file_path (str): The path to the CSV file.\n        \n    Returns:\n        pd.DataFrame: DataFrame containing the loaded data.\n    \"\"\"\n    try:\n        data = pd.read_csv(file_path)\n        return data\n    except FileNotFoundError:\n        print(f\"Error: The file {file_path} was not found.\")\n        return None\n\n# Main analysis function\ndef analyze_data(file_path):\n    \"\"\"\n    Perform data analysis on the dataset.\n    \n    Parameters:\n        file_path (str): The path to the CSV file.\n    \"\"\"\n    # Step 1: Load the data\n    data = load_data(file_path)\n    if data is None:\n        return\n    \n    # Step 2: Preprocess the data (from helper.py)\n    processed_data = preprocess_data(data)\n    \n    # Step 3: Calculate summary statistics (from helper.py)\n    summary_stats = calculate_summary_statistics(processed_data)\n    print(\"Summary Statistics:\")\n    print(summary_stats)\n    \n    # Step 4: Plotting\n    plt.figure(figsize=(10, 6))\n    plt.hist(processed_data['column_of_interest'], bins=30, alpha=0.7)\n    plt.title('Distribution of Column of Interest')\n    plt.xlabel('Value')\n    plt.ylabel('Frequency')\n    plt.show()\n\nif __name__ == '__main__':\n    # Provide the path to your CSV file\n    file_path = 'data.csv'\n    \n    # Run the data analysis\n    analyze_data(file_path)"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "{\"referenced_filepaths\":[{\"filepath\":\"data.csv\"}],\"imported_filepaths\":[{\"import_filepath\":\"helpers.helper\"}]}"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": python_text
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=4095,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "filepath_references",
        "strict": True,
        "schema": {
            "type": "object",
            "required": [
            "referenced_filepaths",
            "imported_filepaths"
            ],
            "properties": {
            "referenced_filepaths": {
                "type": "array",
                "items": {
                "type": "object",
                "required": [
                    "filepath"
                ],
                "properties": {
                    "filepath": {
                    "type": "string"
                    }
                },
                "additionalProperties": False
                }
            },
            "imported_filepaths": {
                "type": "array",
                "items": {
                "type": "object",
                "required": [
                    "import_filepath"
                ],
                "properties": {
                    "import_filepath": {
                    "type": "string",
                    "description": "The path or module of the imported Python file"
                    }
                },
                "additionalProperties": False
                }
            }
            },
            "additionalProperties": False
        }
        }
    }
    )
    text = response.choices[0].message.content
    obj = json.loads(text)

    return obj

def helper_filter_generic_imports(directory):
    # Create a set of all possible module name suffixes from the directory file paths
    module_name_suffixes = set()
    for filepath in directory.keys():
        # Only consider .py files
        if filepath.endswith('.py'):
            # Remove .py extension
            module_path = filepath[:-3]
            # Replace / and \ with . to get module name
            module_name = module_path.replace('/', '.').replace('\\', '.')
            # Split the module name into parts
            parts = module_name.split('.')
            # Generate all possible suffixes
            for i in range(len(parts)):
                suffix = '.'.join(parts[i:])
                module_name_suffixes.add(suffix)

    # Now, for each file in the directory
    for file_info in directory.values():
        referenced = file_info.get('referenced_filepaths', {})
        imported_filepaths = referenced.get('imported_filepaths', [])
        # We will build a new list of imported_filepaths that could be internal
        new_imported_filepaths = []
        for import_entry in imported_filepaths:
            import_filepath = import_entry.get('import_filepath')
            if import_filepath in module_name_suffixes:
                new_imported_filepaths.append(import_entry)
        # Update the imported_filepaths list
        referenced['imported_filepaths'] = new_imported_filepaths
        # Update the file's referenced_filepaths
        file_info['referenced_filepaths'] = referenced

    return directory


# Takes in: python_files_content (obj), file_tree_structure (obj)
# Returns: clean_filetree_object (obj)
def structure_filetree(filetree, content_object):
    reference_tree, file_list = helper_create_summarized_filetree(filetree, content_object)
    restructured_tree_text = ai_structure_filetree(reference_tree, file_list)
    restructured_tree_json = ai_format_filetree_text(restructured_tree_text)
    
    original_to_final_paths = helper_map_paths(filetree, restructured_tree_json)
    
    return restructured_tree_json, original_to_final_paths

def helper_map_paths(original_tree, restructured_tree):
    from collections import defaultdict

    def collect_files(tree, path_so_far, files):
        name = tree['name']
        is_dir = tree['is_directory']
        current_path = path_so_far + [name]
        if is_dir:
            for child in tree.get('children', []):
                collect_files(child, current_path, files)
        else:
            # It's a file
            full_path = '/'.join(current_path)
            files.append({'name': name, 'path': full_path})

    # Collect files from the original tree
    original_files = []
    collect_files(original_tree, [], original_files)
    # Collect files from the restructured tree
    restructured_files = []
    collect_files(restructured_tree, [], restructured_files)

    # Build mappings from file names to paths
    original_file_paths = defaultdict(list)
    for f in original_files:
        original_file_paths[f['name']].append(f['path'])

    restructured_file_paths = defaultdict(list)
    for f in restructured_files:
        restructured_file_paths[f['name']].append(f['path'])

    # Create the mapping from original paths to restructured paths
    original_to_final_paths = {}
    for file_name, original_paths in original_file_paths.items():
        new_paths = restructured_file_paths.get(file_name, [])
        for original_path in original_paths:
            if new_paths:
                # If multiple new paths exist, you may need additional logic to match correctly
                original_to_final_paths[original_path] = new_paths[0]
            else:
                # Handle files that no longer exist in the restructured tree if necessary
                original_to_final_paths[original_path] = None

    return original_to_final_paths

def helper_create_summarized_filetree(filetree, content_object):
    file_list = []

    def process_node(node, path_so_far, is_root=False):
        result = {"name": node["name"]}

        # For the root node, do not include its name in the path
        if is_root:
            new_path = path_so_far
        else:
            new_path = path_so_far + node["name"] if path_so_far else node["name"]

        if node.get("is_directory", False):
            # Process child nodes if they exist
            children = node.get("children", [])
            processed_children = []
            for child in children:
                # For directories, add '/' to path
                child_path_so_far = new_path + '/' if new_path else ''
                processed_child = process_node(child, child_path_so_far, is_root=False)
                if processed_child:
                    processed_children.append(processed_child)
            if processed_children:
                result["children"] = processed_children
        else:
            # Add to file list for non-directory nodes (files)
            file_list.append(new_path)

            # Add summaries for Python files if available
            if node["name"].endswith('.py'):
                content_path = new_path
                if content_path in content_object:
                    content = content_object[content_path]
                    result["file_summary"] = content.get("file_summary")
                    result["function_information"] = content.get("function_information")
        return result

    # Start processing from the root node, setting is_root=True
    summarized_tree = process_node(filetree, "", is_root=True)

    # Return both the summarized filetree and the file list as a string
    file_list_string = "\n".join(file_list)
    return summarized_tree, file_list_string

def ai_structure_filetree(content_json, file_list):
    response = client.chat.completions.create(
    model="o1-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Descriptive JSON: ```({'name': 'weather_sample_directory', 'children': [{'name': 'weather_reading.py', 'file_summary': \"The `weather_reading.py` script is designed to read weather data from CSV files for the years 2018, 2019, and 2020. It includes the `read_csv` function which opens a specified CSV file, reads its content using the CSV reader, and returns the data as a list of rows. It then uses this function to load data from three specified CSV files ('data/weather_2018.csv', 'data/weather_2019.csv', and 'data/weather_2020.csv'), storing the results in the variables `data_2018`, `data_2019`, and `data_2020`. The content of the data for 2018 and 2019 is printed for debugging. This script does not depend on other files in the project and acts independently for data reading purposes.\", 'function_information': [{'function_name': 'read_csv', 'function_description': 'Reads a CSV file from the specified file path and returns its content as a list of rows.'}]}, {'name': 'data_1', 'children': [{'name': 'weather.csv'}]}, {'name': 'weather_processing.py', 'file_summary': \"The `weather_processing.py` script reads CSV files containing weather data and processes this data to calculate and print the average temperature and humidity. It defines a function `process_weather`, which accepts a file path as an argument and reads the weather data, expecting temperatures in the second column and humidity in the third column. The script then calculates the average values of these parameters by summing them up and dividing by the number of records. It prints the results directly without returning them.\\n\\nThe script processes weather data from two files, 'data/weather_2018.csv' and 'data/weather_2019.csv'. The script relies on the `csv` module for handling CSV file operations.\\n\\nNo external functions from other scripts or additional functionalities from the 'utils.py' or 'stats_calculator.py' are used in this script, indicating that it operates independently within its scope.\", 'function_information': [{'function_name': 'process_weather', 'function_description': \"Processes a CSV file containing weather data to calculate and print the average temperature and humidity. The file is specified by 'file_path', and it is expected to have temperature and humidity data in the second and third columns, respectively, with a header that is skipped.\"}]}, {'name': 'utils.py', 'file_summary': 'The `utils.py` file defines a utility function, `to_float(value)`, which attempts to convert an input value to a float. If the conversion is successful, it returns the float value. Otherwise, it returns the string \"N/A\". This function can be useful for handling data conversions where the input might not always be in a numeric format, thus providing a default fallback (\"N/A\") when the conversion fails. Although the file doesn\\'t directly interact with other files, the `to_float` function could be referenced by other scripts in the project for consistent handling of potential numeric conversions, especially when dealing with data entries from CSV files which might not always be clean or well-formatted.', 'function_information': [{'function_name': 'to_float', 'function_description': \"Attempts to convert a given value to a float; returns the float value if successful, otherwise returns 'N/A'.\"}]}, {'name': 'README.txt'}, {'name': 'stats_calculator.py', 'file_summary': 'The `stats_calculator.py` script is designed to calculate and print the minimum and maximum temperatures from a given dataset. It contains two functions: \\n\\n1. `get_min_max(data)`: This function attempts to determine the minimum and maximum temperature from the input data, which it assumes has temperature values in the second element of each row. However, it does not properly convert these values to integers, which could lead to issues if data is not in the expected format.\\n\\n2. `calculate_statistics()`: This is a higher-level function that defines a static dataset directly within the function, calls `get_min_max` to compute the min and max temperatures, and then prints the results. The dataset used in this function has some incomplete entries and misformatted items, which could cause the calculations to behave unexpectedly. Furthermore, this script directly prints results rather than outputting them to a CSV or file, and it does not adequately handle instances where data might be missing or incomplete.\\n\\nThere are no external dependencies within the script itself, but it assumes it can interact with CSV files and real datasets, potentially involving files managed by other scripts like `weather_reading.py` or `weather_processing.py` for acquiring or processing data.', 'function_information': [{'function_name': 'get_min_max', 'function_description': 'Finds and returns the minimum and maximum temperature from a dataset. Assumes the second element in each row of data corresponds to the temperature.'}, {'function_name': 'calculate_statistics', 'function_description': \"Calculates and prints the minimum and maximum temperatures from a dataset. It calls 'get_min_max' function and handles the data directly within the same function.\"}]}, {'name': 'weather_stats_final.csv'}]}```\n\nList of Project Files: ```'weather_reading.py\\ndata_1/weather.csv\\nweather_processing.py\\nutils.py\\nREADME.txt\\nstats_calculator.py\\nweather_stats_final.csv'```\n\nYou are a codebase restructurer. You will be provided a directory JSON that contains the filetree and information about each python file. You should restructure the project using project organization best practices. Please make a new filetree that is maximally organized and intuitive so that a new user could download it and understand what's going on. It is important that the new project contains only and all of the files in the original. File names should remain the same. Please respond with only the new organized filetree. "
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "```\nweather_sample_directory/\n├── data/\n│   ├── raw/\n│   │   └── weather.csv\n│   └── processed/\n│       └── weather_stats_final.csv\n├── src/\n│   ├── weather_reading.py\n│   ├── weather_processing.py\n│   ├── stats_calculator.py\n│   └── utils.py\n├── README.txt\n```"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Descriptive JSON: ```{content_json}```\n\nList of Project Files: ```{file_list}```\n\nYou are a codebase restructurer. You will be provided a directory JSON that contains the filetree and information about each python file. You should restructure the project using project organization best practices. Please make a new filetree that is maximally organized and intuitive so that a new user could download it and understand what's going on. It is important that the new project contains only and all of the files in the original. File names should remain the same. Please respond with only the new organized filetree. "
            }
        ]
        }
    ]
    )
    text = response.choices[0].message.content

    return text

def ai_format_filetree_text(filetree_string):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": "You are an expert file organizer. Given an existing file structure and summaries of each file, propose a new, more intuitive directory structure. It is important that all the file names remain unchanged and that all files are included. ",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "|-- uploads/\n|   |-- selected_few_shot_pipeline/\n|   |   |-- .DS_Store\n|   |   |-- __pycache__/\n|   |   |   |-- api.cpython-312.pyc\n|   |   |   |-- api.cpython-37.pyc\n|   |   |   |-- api.cpython-39.pyc\n|   |   |-- api.py\n|   |   |-- classifier.py\n|   |   |-- classifier_new_data.py\n|   |   |-- errors.csv\n|   |   |-- find_errors.py\n|   |   |-- intercoder_metrics.py\n|   |   |-- metrics.py\n|   |   |-- prompts.json\n|   |   |-- refactor_prompts.py\n|   |   |-- results/\n|   |   |   |-- results.csv\n|   |   |   |-- results_acot_150.csv\n|   |   |   |-- results_split_150.csv\n|   |   |   |-- results_split_150_old_prompts.csv\n|   |   |   |-- results_split_50.csv\n|   |   |   |-- results_split_500_old_prompts.csv\n|   |   |   |-- results_split_50_old_prompts.csv\n|   |   |-- results_split_400.csv\n|   |   |-- results_split_500.csv\n|   |   |-- results_splitcot_400.csv\n|   |   |-- results_test_3_few_4o.csv\n|   |   |-- results_test_ACoT_4o.csv\n|   |   |-- results_test_ind_cot.csv\n|   |   |-- results_test_manyshot_4o.csv\n|   |   |-- results_test_manyshot_optimized_4o.csv\n|   |   |-- results_test_manyshot_optimized_v1_4o.csv\n|   |   |-- results_test_manyshot_optimized_v2_4o.csv\n|   |   |-- results_test_manyshot_precise_seperate_delay_4o.csv\n|   |   |-- results_test_manyshot_precise_system_prompts_stronger_4o.csv\n|   |   |-- results_train_3_few_4o.csv\n|   |   |-- results_train_manyshot_4turbo.csv\n|   |   |-- results_train_manyshot_optimized_4o.csv\n|   |   |-- results_validation_3_few_4o.csv\n|   |   |-- results_validation_manyshot_4o.csv\n|   |   |-- results_validation_manyshot_all_claims_4o.csv\n|   |   |-- results_validation_manyshot_optimized_2check_4o.csv\n|   |   |-- results_validation_manyshot_optimized_4o.csv\n|   |   |-- results_validation_manyshot_optimized_4o_temp1.csv\n|   |   |-- results_validation_manyshot_optimized_v1_4o.csv\n|   |   |-- results_validation_manyshot_precise_seperate_delay_4o.csv\n|   |   |-- results_validation_manyshot_precise_system_prompts_4o.csv\n|   |   |-- results_validation_manyshot_precise_system_prompts_stronger_4o.csv\n|   |   |-- verify_new_prompts.py"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "text": "{\"name\":\"uploads\",\"is_directory\":true,\"children\":[{\"name\":\"selected_few_shot_pipeline\",\"is_directory\":true,\"children\":[{\"name\":\".DS_Store\",\"is_directory\":false,\"children\":[]},{\"name\":\"__pycache__\",\"is_directory\":true,\"children\":[{\"name\":\"api.cpython-312.pyc\",\"is_directory\":false,\"children\":[]},{\"name\":\"api.cpython-37.pyc\",\"is_directory\":false,\"children\":[]},{\"name\":\"api.cpython-39.pyc\",\"is_directory\":false,\"children\":[]}]},{\"name\":\"api.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"classifier.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"classifier_new_data.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"errors.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"find_errors.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"intercoder_metrics.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"metrics.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"prompts.json\",\"is_directory\":false,\"children\":[]},{\"name\":\"refactor_prompts.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"results\",\"is_directory\":true,\"children\":[{\"name\":\"results.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_acot_150.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_150.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_150_old_prompts.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_50.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_500_old_prompts.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_50_old_prompts.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_400.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_500.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_splitcot_400.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_3_few_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_ACoT_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_ind_cot.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_optimized_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_optimized_v1_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_optimized_v2_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_precise_seperate_delay_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_precise_system_prompts_stronger_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_train_3_few_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_train_manyshot_4turbo.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_train_manyshot_optimized_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_3_few_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_all_claims_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_optimized_2check_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_optimized_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_optimized_4o_temp1.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_optimized_v1_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_precise_seperate_delay_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_precise_system_prompts_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_precise_system_prompts_stronger_4o.csv\",\"is_directory\":false,\"children\":[]}]},{\"name\":\"verify_new_prompts.py\",\"is_directory\":false,\"children\":[]}]}]}",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "text": filetree_string,
            "type": "text"
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=2612,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "directory_tree",
        "schema": {
            "type": "object",
            "required": [
            "name",
            "is_directory",
            "children"
            ],
            "properties": {
            "name": {
                "type": "string"
            },
            "children": {
                "type": "array",
                "items": {
                "$ref": "#"
                }
            },
            "is_directory": {
                "type": "boolean"
            }
            },
            "additionalProperties": False
        },
        "strict": True
        }
    }
    )
    text = response.choices[0].message.content
    obj = json.loads(text)

    return obj


# Takes in: python_files_content (obj), file_map_object (obj)
# Returns: python_files_content (obj)
def fix_differences(content_object, filetree_map_object):
    updated_content_object = helper_determine_replacements(content_object, filetree_map_object)
    updated_content_object = helper_update_code(updated_content_object)
    return updated_content_object

def helper_determine_replacements(content_object, filetree_map_object):
    # Define the project root directory (base directory)
    project_root = "/Users/typham-swann/Desktop/research_code_app"
    base_dir_name = "python_logic_tester"
    base_dir = os.path.join(project_root, base_dir_name)

    # Build a mapping from absolute original paths to absolute new paths
    original_to_new_paths = {}
    for original_rel_path, new_rel_path in filetree_map_object.items():
        original_abs_path = os.path.normpath(os.path.join(project_root, original_rel_path))
        new_abs_path = os.path.normpath(os.path.join(project_root, new_rel_path))
        original_to_new_paths[original_abs_path] = new_abs_path

    # Initialize the updated content object
    updated_content_object = {}

    # Process each file in the content object
    for file_rel_path, file_data in content_object.items():
        # Get original and new absolute paths of the file
        original_file_abs_path = os.path.normpath(os.path.join(base_dir, file_rel_path))
        new_file_abs_path = original_to_new_paths.get(original_file_abs_path, original_file_abs_path)

        # Initialize the updated references list
        updated_references = []

        # Get referenced filepaths
        referenced_filepaths = file_data.get('referenced_filepaths', {})
        for ref_type in ['referenced_filepaths', 'imported_filepaths']:
            ref_list = referenced_filepaths.get(ref_type, [])
            for ref in ref_list:
                original_ref_path = ref.get('filepath')
                if not original_ref_path:
                    continue

                # Normalize the original referenced path
                original_ref_abs_path = os.path.normpath(original_ref_path)
                if not original_ref_abs_path.startswith(project_root):
                    # If the path is not under project_root, skip it
                    continue

                # Get the relative path from project root
                original_ref_rel_path = os.path.relpath(original_ref_abs_path, project_root)
                original_ref_abs_path = os.path.join(project_root, original_ref_rel_path)

                # Get the new absolute path from the filetree map
                new_ref_abs_path = original_to_new_paths.get(original_ref_abs_path, original_ref_abs_path)

                # Compute the new relative path from the new file location to the new referenced file location
                new_relative_path = os.path.relpath(new_ref_abs_path, os.path.dirname(new_file_abs_path))

                # Add to updated_references
                updated_references.append({
                    'original_filepath': original_ref_path,
                    'new_filepath': new_relative_path
                })

        # Add 'updated_references' to the file data
        updated_file_data = dict(file_data)
        updated_file_data['updated_references'] = updated_references

        # Update the content object with the new data
        updated_content_object[file_rel_path] = updated_file_data

    return updated_content_object

def helper_update_code(content_object):
    updated_content_object = {}

    for file_rel_path, file_data in content_object.items():
        code = file_data.get('code', '')
        updated_references = file_data.get('updated_references', [])

        # Make a copy of the code to perform replacements
        updated_code = code

        for ref in updated_references:
            original_filepath = ref.get('original_filepath')
            new_filepath = ref.get('new_filepath')

            if not original_filepath or not new_filepath:
                continue

            # Escape special characters in filepaths for regex
            escaped_original_filepath = re.escape(original_filepath)

            # Use regex to replace all occurrences of the old filepath with the new filepath
            updated_code = re.sub(escaped_original_filepath, new_filepath, updated_code)

        # Update the code in the file data
        updated_file_data = dict(file_data)
        updated_file_data['code'] = updated_code

        # Update the content object with the new file data
        updated_content_object[file_rel_path] = updated_file_data

    return updated_content_object


# Takes in: python_files_content (obj), filetree (obj)
# Returns: notebook (obj)
def generate_notebook(content_object, filetree_object):
    notebook_plan = ai_helper_plan_notebook(content_object, filetree_object)
    notebook_string = ai_helper_create_notebook(content_object, notebook_plan)
    notebook_object = ai_helper_create_notebook_object(notebook_string)
    return notebook_object

def ai_helper_plan_notebook(content_object, filetree_object):
    content_object = helper_filter_code_fields(content_object)

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are tasked with planning a turnkey Jupiter notebook for a codebase directory. You must understand the project and create a comprehensive plan for what an ideal notebook would look like for someone who has never before seen the project but would like to get started. You will be provided a filetree and a description of the python files within the project. Your plan should mention all necessary code and functions that must be included in the notebook, including mentioning which files the functions should come from. You do not need to provide the actual code, as the creator of the notebook will have the project code; simply, you need to say which functions should be used and from which files."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Filetree Object: ```{\n    \"name\": \"nlp_project\",\n    \"children\": [\n        {\n            \"name\": \"data\",\n            \"children\": [\n                {\n                    \"name\": \"raw\",\n                    \"children\": [\n                        {\n                            \"name\": \"reviews.csv\",\n                            \"children\": [],\n                            \"is_directory\": false\n                        }\n                    ],\n                    \"is_directory\": true\n                },\n                {\n                    \"name\": \"processed\",\n                    \"children\": [\n                        {\n                            \"name\": \"cleaned_reviews.csv\",\n                            \"children\": [],\n                            \"is_directory\": false\n                        }\n                    ],\n                    \"is_directory\": true\n                }\n            ],\n            \"is_directory\": true\n        },\n        {\n            \"name\": \"models\",\n            \"children\": [\n                {\n                    \"name\": \"sentiment_model.pkl\",\n                    \"children\": [],\n                    \"is_directory\": false\n                }\n            ],\n            \"is_directory\": true\n        },\n        {\n            \"name\": \"scripts\",\n            \"children\": [\n                {\n                    \"name\": \"preprocessing.py\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"modeling.py\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"visualization.py\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"data_loading.py\",\n                    \"children\": [],\n                    \"is_directory\": false\n                }\n            ],\n            \"is_directory\": true\n        },\n        {\n            \"name\": \"docs\",\n            \"children\": [\n                {\n                    \"name\": \"project_overview.md\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"function_descriptions.json\",\n                    \"children\": [],\n                    \"is_directory\": false\n                }\n            ],\n            \"is_directory\": true\n        },\n        {\n            \"name\": \"output\",\n            \"children\": [\n                {\n                    \"name\": \"evaluation_report.txt\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"word_cloud.png\",\n                    \"children\": [],\n                    \"is_directory\": false\n                }\n            ],\n            \"is_directory\": true\n        }\n    ],\n    \"is_directory\": true\n}```\n\nPython Content: ```{\n    \"nlp_project/preprocessing.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"tokenize_text\",\n                \"function_description\": \"Splits the input text into tokens (words) using the nltk library and returns a list of tokens.\"\n            },\n            {\n                \"function_name\": \"remove_stopwords\",\n                \"function_description\": \"Removes common English stopwords from the token list using a predefined stopword list from the nltk library.\"\n            },\n            {\n                \"function_name\": \"stem_tokens\",\n                \"function_description\": \"Applies stemming to the token list using the PorterStemmer from the nltk library, returning the stemmed version of each token.\"\n            }\n        ],\n        \"file_summary\": \"The `preprocessing.py` script handles text preprocessing tasks essential for preparing text data for sentiment analysis. It leverages the nltk library for natural language processing functions. The script defines three main functions:\\n\\n1. `tokenize_text(text)`: This function takes an input string and splits it into tokens (words), returning the list of tokens for further processing.\\n\\n2. `remove_stopwords(tokens)`: Removes common English stopwords from the token list to retain only the most meaningful words. It uses a stopword list from nltk for this purpose.\\n\\n3. `stem_tokens(tokens)`: Applies stemming to reduce words to their base or root forms using the PorterStemmer from nltk. This is useful for standardizing words and improving model performance.\\n\\nThe preprocessing steps in this script are crucial for cleaning and transforming text data into a suitable format for sentiment analysis models, ensuring that only relevant information is passed on for further modeling steps.\"\n    },\n    \"nlp_project/modeling.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"train_sentiment_model\",\n                \"function_description\": \"Trains a sentiment analysis model using a TF-IDF vectorizer and a Logistic Regression classifier on the provided training data.\"\n            },\n            {\n                \"function_name\": \"predict_sentiment\",\n                \"function_description\": \"Generates sentiment predictions (positive, negative, or neutral) for input text data using the trained sentiment analysis model.\"\n            },\n            {\n                \"function_name\": \"evaluate_model\",\n                \"function_description\": \"Evaluates the trained sentiment analysis model using test data and calculates accuracy, precision, and recall scores.\"\n            }\n        ],\n        \"file_summary\": \"The `modeling.py` script is dedicated to building and evaluating a sentiment analysis model. It utilizes scikit-learn's Logistic Regression classifier and TF-IDF vectorizer for text feature extraction and model training. The script includes three main functions:\\n\\n1. `train_sentiment_model(X_train, y_train)`: This function takes input features (`X_train`) and target labels (`y_train`) and trains a sentiment analysis model using a TF-IDF vectorizer to transform text data into numerical form. It returns the trained model.\\n\\n2. `predict_sentiment(model, X_test)`: Uses the trained model to predict the sentiment of new input data (`X_test`), returning the predictions as an array of sentiment labels (e.g., positive, negative, or neutral).\\n\\n3. `evaluate_model(model, X_test, y_test)`: Evaluates the model’s performance using the test set, `X_test` and `y_test`, calculating key metrics like accuracy, precision, and recall scores.\\n\\nThe script is designed to support the full modeling pipeline, from training to evaluation, and works with preprocessed text data. It plays a crucial role in developing and fine-tuning the sentiment analysis model.\"\n    },\n    \"nlp_project/visualization.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"plot_sentiment_distribution\",\n                \"function_description\": \"Plots a bar chart showing the distribution of sentiments (positive, negative, neutral) in the dataset using Matplotlib.\"\n            },\n            {\n                \"function_name\": \"plot_word_cloud\",\n                \"function_description\": \"Generates and displays a word cloud visualization of the most frequent words in the text data using the WordCloud library.\"\n            }\n        ],\n        \"file_summary\": \"The `visualization.py` script contains functions for visualizing text data and model outputs. It includes two main functions:\\n\\n1. `plot_sentiment_distribution(data)`: This function takes a dataset containing sentiment labels (positive, negative, neutral) and plots a bar chart showing the distribution of these labels using Matplotlib.\\n\\n2. `plot_word_cloud(text)`: Generates a word cloud visualization from the input text data, highlighting the most frequently occurring words. It uses the WordCloud library for this task.\\n\\nThese visualizations are essential for understanding the characteristics of the dataset and the outputs of the sentiment analysis model. The script is designed to provide quick, informative visual feedback during the development and evaluation of the sentiment analysis pipeline.\"\n    },\n    \"nlp_project/data_loading.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"load_dataset\",\n                \"function_description\": \"Loads the text dataset from a CSV file given by 'filepath' into a pandas DataFrame.\"\n            },\n            {\n                \"function_name\": \"split_data\",\n                \"function_description\": \"Splits the dataset into training and test sets, returning the features and labels for each split.\"\n            },\n            {\n                \"function_name\": \"save_processed_data\",\n                \"function_description\": \"Saves the processed dataset to a CSV file at the specified 'filepath' without the index.\"\n            }\n        ],\n        \"file_summary\": \"The `data_loading.py` script manages the loading, splitting, and saving of datasets used in the NLP project. It uses pandas for handling CSV data and defines three main functions:\\n\\n1. `load_dataset(filepath)`: This function reads a CSV file specified by `filepath` into a pandas DataFrame, which serves as the input for the NLP pipeline.\\n\\n2. `split_data(data)`: Splits the DataFrame into training and test sets, returning the features (text) and labels (sentiments) for each set. This is essential for training and evaluating the sentiment analysis model.\\n\\n3. `save_processed_data(data, filepath)`: Saves the processed DataFrame to a CSV file at the specified `filepath`, excluding the index to ensure compatibility with other scripts in the pipeline.\\n\\nThe script serves as the foundation for data management in the NLP project, ensuring that text data is properly loaded, split, and saved for subsequent processing and modeling tasks.\"\n    }\n}\n```"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "Here's a detailed plan for creating a comprehensive Jupyter notebook for the current NLP project. The goal is to guide a new user through the entire workflow, from data loading and preprocessing to modeling, evaluation, and visualization.\n\n### Notebook Structure and Content\n\n1. **Notebook Setup**\n   - **Introduction**\n     - Brief overview of the project, explaining the goal (e.g., performing sentiment analysis on text data) and the steps involved.\n   - **Environment Setup**\n     - Import necessary libraries (e.g., pandas, nltk, sklearn, matplotlib, etc.).\n\n2. **Data Loading**\n   - **Load Dataset**\n     - Use `load_dataset(filepath)` from `data_loading.py` to load the raw data from `data/raw/reviews.csv` into a pandas DataFrame.\n     - Display the first few rows of the dataset to understand its structure.\n\n3. **Data Preprocessing**\n   - **Text Tokenization**\n     - Use `tokenize_text(text)` from `preprocessing.py` to split input text into tokens.\n   - **Stopwords Removal**\n     - Use `remove_stopwords(tokens)` from `preprocessing.py` to filter out common stopwords and prepare the text for analysis.\n   - **Stemming**\n     - Use `stem_tokens(tokens)` from `preprocessing.py` to perform stemming and reduce words to their root form.\n\n4. **Data Splitting**\n   - **Training and Test Sets**\n     - Utilize `split_data(data)` from `data_loading.py` to divide the preprocessed data into training and test sets.\n     - Display the dimensions of the resulting datasets to verify the split.\n\n5. **Model Training**\n   - **Train Model**\n     - Use `train_sentiment_model(X_train, y_train)` from `modeling.py` to train a sentiment analysis model on the training data.\n     - Provide an overview of the model's basic configuration and feature extraction method (e.g., TF-IDF).\n\n6. **Model Evaluation**\n   - **Predict Sentiments**\n     - Use `predict_sentiment(model, X_test)` from `modeling.py` to generate sentiment predictions on the test set.\n   - **Evaluate Model**\n     - Use `evaluate_model(model, X_test, y_test)` from `modeling.py` to calculate and display model performance metrics such as accuracy, precision, and recall.\n\n7. **Results Visualization**\n   - **Sentiment Distribution**\n     - Use `plot_sentiment_distribution(data)` from `visualization.py` to plot a bar chart of sentiment distribution in the dataset.\n   - **Word Cloud Visualization**\n     - Use `plot_word_cloud(text)` from `visualization.py` to generate word clouds for visual representation of the most frequent words.\n\n8. **Saving Processed Data**\n   - **Save Data**\n     - Use `save_processed_data(data, filepath)` from `data_loading.py` to save the processed dataset to `data/processed/cleaned_reviews.csv`.\n\n9. **Conclusion and Next Steps**\n   - Summarize key findings, insights gained, and possible further exploration.\n   - Provide suggestions for extending the project or improving model performance, such as experimenting with different algorithms or preprocessing techniques.\n\n---\n\nThis structure ensures a logical flow that guides the user through understanding and running the project's essential components while keeping the notebook organized and educational. Each step should include appropriate comments, explanations, and visualization outputs where applicable to enhance understanding and engagement."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Filetree Object: ```{filetree_object}```\n\nPython Content: ```{content_object}```"
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "text"
    }
    )
    notebook_plan = response.choices[0].message.content
    return notebook_plan

def ai_helper_get_code_json(content_object, notebook_plan):
    filepaths = list(content_object.keys())
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You will be provided a plan for creating a notebook, and a project directory JSON. Please create a list of files necessary to create the notebook."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Plan: ```{notebook_plan}```\n\nPython Content JSON: ```{content_object}```"
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "relevent_pilepaths",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
            "filepaths": {
                "type": "array",
                "description": "List of relevant filepaths",
                "items": {
                "type": "string",
                "enum": filepaths
                }
            }
            },
            "required": [
            "filepaths"
            ],
            "additionalProperties": False
        }
        }
    }
    )

    needed_files = response.choices[0].message.content
    return needed_files

def ai_helper_create_notebook(content_object, notebook_plan):
    content_object = helper_filter_code_fields(content_object, ai_helper_get_code_json(content_object, notebook_plan))
    response = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "text": f"Python Contents Object: ```{content_object}```\n\nNotebook Plan: ```{notebook_plan}```\n\nPlease create a full jupiter notebook using the provided code and plan. You response should work entirely standalone as a turnkey notebook for the directory and be formatted in a way that it can directly be converted in a ipynb file. Your audience is another researcher who is picking up the directory to add to. The notebook should not be chatty or educational but straightforward and informative. Please make it clear where the code and markdown sections are",
            "type": "text"
            }
        ]
        }
    ]
    )

    notebook_string = response.choices[0].message.content
    return notebook_string

def ai_helper_create_notebook_object(notebook_string):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are in charge of formatting a python notebook. You will receive a string containing the contents of a notebook where the code and markdown are specified. Please convert it into the format for pynb. "
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Notebook String: ```{notebook_string}```"
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=4095,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "ipynb_notebook",
        "schema": {
            "type": "object",
            "required": [
            "metadata",
            "nbformat",
            "nbformat_minor",
            "cells"
            ],
            "properties": {
            "cells": {
                "type": "array",
                "items": {
                "type": "object",
                "required": [
                    "cell_type",
                    "source",
                    "execution_count",
                    "outputs"
                ],
                "properties": {
                    "source": {
                    "type": "string"
                    },
                    "outputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                        "output_type",
                        "text",
                        "name",
                        "execution_count",
                        "ename",
                        "evalue",
                        "traceback"
                        ],
                        "properties": {
                        "data": {
                            "type": "object",
                            "additionalProperties": False
                        },
                        "name": {
                            "type": "string"
                        },
                        "text": {
                            "type": "string"
                        },
                        "ename": {
                            "type": "string"
                        },
                        "evalue": {
                            "type": "string"
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": False
                        },
                        "traceback": {
                            "type": "array",
                            "items": {
                            "type": "string"
                            }
                        },
                        "output_type": {
                            "type": "string"
                        },
                        "execution_count": {
                            "type": [
                            "integer",
                            "null"
                            ]
                        }
                        },
                        "additionalProperties": False
                    }
                    },
                    "metadata": {
                    "type": "object",
                    "additionalProperties": False
                    },
                    "cell_type": {
                    "enum": [
                        "code",
                        "markdown"
                    ],
                    "type": "string"
                    },
                    "execution_count": {
                    "type": [
                        "integer",
                        "null"
                    ]
                    }
                },
                "additionalProperties": False
                }
            },
            "metadata": {
                "type": "object",
                "required": [
                "kernel_info",
                "language_info"
                ],
                "properties": {
                "kernel_info": {
                    "type": "object",
                    "required": [
                    "name"
                    ],
                    "properties": {
                    "name": {
                        "type": "string"
                    }
                    },
                    "additionalProperties": False
                },
                "language_info": {
                    "type": "object",
                    "required": [
                    "name",
                    "version",
                    "codemirror_mode"
                    ],
                    "properties": {
                    "name": {
                        "type": "string"
                    },
                    "version": {
                        "type": "string"
                    },
                    "codemirror_mode": {
                        "type": "string"
                    }
                    },
                    "additionalProperties": False
                }
                },
                "additionalProperties": False
            },
            "nbformat": {
                "type": "integer"
            },
            "nbformat_minor": {
                "type": "integer"
            }
            },
            "additionalProperties": False
        },
        "strict": True
        }
    }
    )
    notebook_object = json.loads(response.choices[0].message.content)
    return notebook_object

def helper_filter_code_fields(content_object, files_to_keep=None):
    # If files_to_keep is None, set it as an empty list to filter out all 'code' fields
    if files_to_keep is None:
        files_to_keep = []

    filtered_content = {}
    for filepath, details in content_object.items():
        # Copy the details dictionary to avoid mutating the original content_object
        filtered_details = details.copy()

        # If the filepath is not in the files_to_keep list, remove the 'code' field
        if filepath not in files_to_keep:
            filtered_details.pop("code", None)
        
        # Remove 'referenced_filepaths' and 'updated_references' for all files
        filtered_details.pop("referenced_filepaths", None)
        filtered_details.pop("updated_references", None)

        filtered_content[filepath] = filtered_details
    
    return filtered_content


# Takes in: content_object, filetree_object, notebook_content_json (str)
# Returns: clean filetree (obj)
def construct_directory(original_dir, content_object, new_filetree, output_dir, notebook_object):
    paths_map = helper_create_path_mapping(original_dir, new_filetree)
    helper_build_directory(original_dir, content_object, new_filetree, paths_map, output_dir)
    helper_add_notebook(notebook_object, output_dir, new_filetree)  # Pass new_filetree to locate project root

def helper_create_path_mapping(base_dir, filetree):
    def get_existing_files(directory):
        """Returns list of all files (not directories) under the given directory."""
        existing_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                # Get full absolute path
                full_path = os.path.join(root, file)
                # Get path relative to base_dir
                rel_path = os.path.relpath(full_path, directory)
                existing_files.append(rel_path)
        return existing_files

    def get_filetree_paths(node, current_path=""):
        """Extracts all file paths from the filetree JSON structure."""
        paths = []
        new_path = os.path.join(current_path, node["name"])

        if not node["is_directory"]:
            # Do not include base_dir in the path
            paths.append(new_path)

        for child in node.get("children", []):
            paths.extend(get_filetree_paths(child, new_path))

        return paths

    def find_matching_file(original_file, filetree_paths):
        """
        Finds the matching new path for an original file based on filename.
        Returns None if no match is found.
        """
        filename = os.path.basename(original_file)
        matches = [p for p in filetree_paths if os.path.basename(p) == filename]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # If multiple matches, try to match based on file type/extension
            exact_matches = [p for p in matches if p.endswith(os.path.splitext(filename)[1])]
            return exact_matches[0] if exact_matches else matches[0]
        return None

    # Get list of existing files with relative paths
    existing_files = get_existing_files(base_dir)

    # Get list of paths from filetree (relative paths)
    filetree_paths = get_filetree_paths(filetree)

    # Create mapping
    path_mapping = {}
    for orig_path in existing_files:
        new_path = find_matching_file(orig_path, filetree_paths)
        if new_path:
            path_mapping[orig_path] = new_path

    return path_mapping

# Takes in: content_object, filetree_object, notebook_content_json (str)
# Returns: clean filetree (obj)
def construct_directory(original_dir, content_object, new_filetree, output_dir, notebook_object=None):
    paths_map = helper_create_path_mapping(original_dir, new_filetree)
    helper_build_directory(original_dir, content_object, new_filetree, paths_map, output_dir)
    if notebook_object is not None:
        helper_add_notebook(notebook_object, output_dir, new_filetree)  # Pass new_filetree to locate project root

def helper_create_path_mapping(base_dir, filetree):
    def get_existing_files(directory):
        """Returns list of all files (not directories) under the given directory."""
        existing_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                # Get full absolute path
                full_path = os.path.join(root, file)
                # Get path relative to base_dir
                rel_path = os.path.relpath(full_path, directory)
                existing_files.append(rel_path)
        return existing_files

    def get_filetree_paths(node, current_path=""):
        """Extracts all file paths from the filetree JSON structure."""
        paths = []
        new_path = os.path.join(current_path, node["name"])

        if not node["is_directory"]:
            # Do not include base_dir in the path
            paths.append(new_path)

        for child in node.get("children", []):
            paths.extend(get_filetree_paths(child, new_path))

        return paths

    def find_matching_file(original_file, filetree_paths):
        """
        Finds the matching new path for an original file based on filename.
        Returns None if no match is found.
        """
        filename = os.path.basename(original_file)
        matches = [p for p in filetree_paths if os.path.basename(p) == filename]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # If multiple matches, try to match based on file type/extension
            exact_matches = [p for p in matches if p.endswith(os.path.splitext(filename)[1])]
            return exact_matches[0] if exact_matches else matches[0]
        return None

    # Get list of existing files with relative paths
    existing_files = get_existing_files(base_dir)

    # Get list of paths from filetree (relative paths)
    filetree_paths = get_filetree_paths(filetree)

    # Create mapping
    path_mapping = {}
    for orig_path in existing_files:
        new_path = find_matching_file(orig_path, filetree_paths)
        if new_path:
            path_mapping[orig_path] = new_path

    return path_mapping

def helper_build_directory(old_directory_path, content_object, filetree_object, paths_map, output_directory_path):
    # Function to recursively create directories as per the filetree object
    def create_directories(tree_node, current_path):
        if tree_node.get('is_directory', False):
            dir_name = tree_node['name']
            dir_path = os.path.join(current_path, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            for child in tree_node.get('children', []):
                create_directories(child, dir_path)
    
    # Create the new directory structure in the output directory
    create_directories(filetree_object, output_directory_path)
    
    # Process each file in the paths map
    for old_file_rel_path, new_file_rel_path in paths_map.items():
        old_file_abs_path = os.path.join(old_directory_path, old_file_rel_path)
        new_file_abs_path = os.path.join(output_directory_path, new_file_rel_path)

        # Ensure the directory for the new file exists
        new_file_dir = os.path.dirname(new_file_abs_path)
        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)

        if new_file_abs_path.endswith('.py'):
            # It's a Python file; replace its content with the code from the content object
            filename = os.path.basename(old_file_rel_path)
            code_info = content_object.get(filename)
            if code_info:
                code_content = code_info.get('code', '')
                with open(new_file_abs_path, 'w') as file:
                    file.write(code_content)
            else:
                print(f"Warning: Code for {filename} not found in the content object. Copying original file.")
                print(filename)
                if os.path.exists(old_file_abs_path):
                    shutil.copy2(old_file_abs_path, new_file_abs_path)
                else:
                    print(f"Error: Original file {old_file_abs_path} does not exist.")
        else:
            # For non-Python files, copy them to the new location without modification
            if os.path.exists(old_file_abs_path):
                shutil.copy2(old_file_abs_path, new_file_abs_path)
            else:
                print(f"Error: Original file {old_file_abs_path} does not exist.")

def helper_add_notebook(notebook_object, output_directory_path, filetree_object):
    """Adds a starter notebook to the root of the project directory."""
    # The root of the project is the top-level directory in the filetree
    project_root_name = filetree_object['name']
    project_root_path = os.path.join(output_directory_path, project_root_name)

    # Ensure the project root directory exists
    if not os.path.exists(project_root_path):
        os.makedirs(project_root_path)

    # Path to place the notebook
    notebook_path = os.path.join(project_root_path, 'starter_notebook.ipynb')

    # Create the notebook
    nb = nbformat.from_dict(notebook_object)
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    

# Takes in: directory (str)
# Returns: clean filetree (obj)
# run_pipeline.py

def run_full_pipeline_generator(input_directory, output_directory, generate_notebook=True):
    yield "(1/10) Importing files..."
    time.sleep(0.1)  # Small delay to flush each yield
    python_files_content, file_tree_structure = handle_files(input_directory)

    yield "(2/10) Commenting code..."
    time.sleep(0.1)
    python_files_content = comment_code(python_files_content)

    yield "(3/10) Understanding functions..."
    time.sleep(0.1)
    python_files_content = summarize_functions(python_files_content)

    yield "(4/10) Understanding project structure..."
    time.sleep(0.1)
    python_files_content = summarize_files(python_files_content)

    yield "(5/10) Understanding nesting..."
    time.sleep(0.1)
    original_python_files_content = identify_references(python_files_content)

    yield "(6/10) Generating organized filetree..."
    time.sleep(0.1)
    organized_file_tree, filetree_map = structure_filetree(file_tree_structure, original_python_files_content)

    yield "(7/10) Fixing refactoring changes..."
    time.sleep(0.1)
    python_files_content = fix_differences(original_python_files_content, filetree_map)

    if generate_notebook:
        yield "(8/10) Generating starter notebook (the slow part)..."
        time.sleep(0.1)
        notebook_content = generate_notebook(original_python_files_content, organized_file_tree)
    else:
        notebook_content = None

    yield "(9/10) Outputting files..."
    time.sleep(0.1)
    construct_directory(input_directory, python_files_content, organized_file_tree, output_directory, notebook_content)

    yield "(10/10) All done"



# run_pipeline(sample_directory, output_dir)
