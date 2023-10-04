# Hate Speech Detection using Attention Mechanisms on Hinglish Tweets

# **Project Overview**
This project focuses on Hate Speech Detection in Hinglish Tweets, a fusion of English and Hindi. Leveraging Bahadanau's attention mechanism in Transformer Models and also applying various embeddings, including Word2Vec, Doc2Vec, FastText on various Classical Machine Learning Algorithms and Deep Learning Models (CNN, LSTM, RNN, Bi-LSTM), the goal is to create robust models for identifying hate speech in a linguistically diverse dataset.

## Dependencies
- Ensure you have the necessary dependencies installed.
- Refer to the requirements.txt file for version details.
```
pip install -r requirements.txt
```


## Python Packages Used
In this section, I include all the necessary dependencies needed to reproduce the project, so that the reader can install them before replicating the project. I categorize the long list of packages used as - 
- **General Purpose:** `warnings, shutil, glob, tqdm, string, os`
- **Data Manipulation:** `numpy, nlkt, pandas, gensim`
- **Data Visualization:** `matplotlib, wordcloud`
- **Machine Learning:** `keras, tensorflow, tensorflow_hub, sklearn`

## Dataset and Preprocessing
- **Dataset Characteristics:**
- Hinglish Tweets with English text but Hindi language elements.
- Hate speech tweets: 49%, Non-hate speech tweets: 51%.
- **Preprocessing Steps:**
- Regex-based cleaning to remove unwanted characters.
- Elimination of usernames, hashtags, URLs, and special characters.
- Word cloud analysis for insights into hate speech terms.

## Word Embeddings
- **Word2Vec, Doc2Vec, FastText:**
- Applied to classical ML models (Logistic Regression, SVM, Random Forest).
- Integration into deep learning models (Conv1D, RNN, LSTM, Bi-LSTM).
- Exploration of character n-gram models in FastText.

# Transformer Models
## **1. Simple Transformer Models**
- Token and positional embeddings with multihead attention mechanism.
- Global Average Pooling, dropout, dense layers, and sigmoid activation.

## **2. Encoder-Decoder Architecture Transformer Models:**
- LSTM-based encoder and decoder architecture.
- Applied to hate speech detection.

## **3. Encoder-Decoder Architecture with Bahadanau's Attention Mechanism:**
- Attention mechanism integration using Bahadanau's method.
- Achieved a 69% accuracy on the test dataset.

## **How To Use:**
### 1. Data Preprocessing:
- **Initial Cleaning:**
Applied regex for removing unwanted characters.
**Usernames and Hashtags:**
- Removed usernames and associated hashtags from tweets.
**URLs and Special Characters:**
- Eliminated URLs and other special characters from the tweets.
**Tokenization:**
- Split tweets into tokens based on spaces.
**Word2Vec Tokenization:**
- Applied Word2Vec to the tokenized sentences in the dataset.
### **2. Word Embeddings and Traditional ML Models:**
**Tokenization and Splitting:**
- Tokenized sentences and split the dataset into training and test sets.
**OOV Tokens Assignment:**
- Provided out-of-vocabulary (OOV) tokens to the tokenizer, fitted it to the training dataset, and obtained tokens.
**Text to Sequences and Padding:**
- Converted texts to sequences and added padding to accommodate the maximum length of sentences.
**Embedding Matrix Preparation:**
- Prepared the embedding matrix for the Embedding layer by deciding the embedding dimensions and technique (e.g., Word2Vec).
- Created an embedding matrix of zeros with dimensions (length of vocab, chosen embedding dimension).
**Text into Sequences Conversion:**
- Converted text into sequences.
**Neural Network Training:**
- Trained Neural Networks (Conv1D, RNN, LSTM, Bi-LSTM) on the processed dataset.

**FastText Application in Models**
Character n-gram Models:
- Represented each word as a bag of character n-grams.
- Considered character n-grams of length 3 to 6 for each word.

### 3. Transformer Attention Models Preprocessing:
**1. Simple Transformer Models with Multihead Attention Mechanism**
- Token and Positional Embeddings:
- Created token and positional embeddings for Transformer inputs.
- Transformer Model Layers Preparation:
- Prepared layers for the Transformer model with positional and token embeddings, along with attention in the Transformer block.
- Applied Global Average Pooling, dropout, dense layers, and a final dropout, followed by output through the sigmoid activation function.

**2. Encoder-Decoder Architecture Transformer Models**
- Encoder-Decoder Architecture:
- Applied an encoder-decoder architecture using LSTM in both the encoder and decoder.
- Used an embedding matrix generated from the Word2Vec model.

**3. Encoder-Decoder Architecture with Bahadanau's Attention Mechanism**
- Simple Attention Mechanism:
- Used Bahadanau's attention mechanism with a simple attention layer.
- Achieved an accuracy of 69% on the test dataset.

## Results:
- Achieved 69% accuracy with the Encoder-Decoder Architecture using Bahadanau's Attention Mechanism.

## Contact
- Connect with the project creator:
- LinkedIn: https://www.linkedin.com/in/srivatsank01/
- Email: srivatsank.dl@gmail.com
- Google Scholar - https://scholar.google.com/citations?user=vIwFiwgAAAAJ&hl=en

# License
For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).
