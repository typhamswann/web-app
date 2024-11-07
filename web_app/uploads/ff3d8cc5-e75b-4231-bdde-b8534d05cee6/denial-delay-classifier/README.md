# Denial Delay Classifier

This project is a classifier for detecting climate denial and delay.

## Installation guide

Clone the repository
```console
git clone https://github.com/willkattrup/denial-delay-classifier.git
```

Setup the virtual environment to install dependencies
```console
python -m venv venv
```

Activate environment
```console
source venv/bin/activate 
```

Install the project
```console
pip install .
```

Set up your API Key
```console
export OPENAI_API_KEY="your api key here"
```

Run the classifier script
## Usage guide

Use classify to input a prompt/paragraphs that needs to be classified as denial/delay/no claim.

Use prompts.py to further prompt-tune the classifier.