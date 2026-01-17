# Cognitive-Distortion-Detection-Project

cdts model setup guide

this project uses machine learning (tf-idf and svm) to detect cognitive distortions in text, such as "all-or-nothing thinking" or "overgeneralization."

1. prerequisites
 
make sure you have python installed on your system.

3. install dependencies

open your terminal or command prompt and run the following command to install the required libraries:

pip install pandas numpy scikit-learn nltk

5. download nltk data

the code uses a lemmatizer and tokenizer. you need to download the necessary resource files by running this small script:

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

4. prepare your data

ensure your dataset file is named cbt_df.csv and is located in the same folder as your python script. your csv should have at least these two columns:

negative_thought: the text sample.

distortions: the label.

5. how to run

save your code as cdts_model.py.

run the script:

python cdts_model.py

the script will:

load and preprocess the data.

train the svm model.

show you a comparison between your results and the target paper results.

ask if you want to save the model (y/n).

ask if you want to test custom thoughts (y/n).

6. project structure

cdts_model.py: the primary script containing the cdts_model class.

cbt_df.csv: the dataset.

cdts_model.pkl: the saved model file (generated after training).
