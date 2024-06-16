# Text-Generation-Using-T5

This project demonstrates the use of Transformers for text generation using the T5 model. The project includes the necessary code for training the model on a custom dataset and generating new text.

## Project Structure

Text-Generation-Using-T5/
│
├── .gitignore
├── README.md
├── data/
│   └── input.txt  # Text file containing input data
├── model/
│   └── t5  # Directory for saving the fine-tuned model
├── generate.py
├── requirements.txt
├── train.py
└── results/
    └── generated_texts.txt  # Generated texts will be saved here


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Text-Generation-Using-T5.git
   cd Text-Generation-Using-T5
2. Install the dependencies
   ```bash
   pip install -r requirements.txt
3. To run the model
   ```bash
   python train.py
4. To generate the text
   ```bash
   python generate.py
