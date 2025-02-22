Sentiment Analysis of Customer Reviews

📌 Project Overview

This project builds a Machine Learning model to classify customer reviews as positive or negative using Natural Language Processing (NLP) techniques. The dataset used is the IMDb Reviews dataset, and the model is trained using Logistic Regression.

📂 Dataset

Dataset Used: IMDb Reviews (or any public review dataset)

Source: https://ai.stanford.edu/~amaas/data/sentiment/

Training Data: 25,000 labeled reviews (positive & negative)

Testing Data: 25,000 labeled reviews

🔍 Steps Followed

1️⃣ Data Collection

Loaded IMDb dataset (both positive and negative reviews).

2️⃣ Data Preprocessing

Converted text to lowercase.

Removed HTML tags, numbers, and punctuation.

Tokenized the text and removed stopwords.

Lemmatized words to their root forms.

3️⃣ Feature Extraction

Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features.

Why TF-IDF over Bag of Words?

TF-IDF gives more importance to unique words.

Helps reduce the influence of common words.

Improves accuracy compared to a simple Bag of Words model.

4️⃣ Model Selection & Training

Chose Logistic Regression for classification.

Performed Hyperparameter Tuning using GridSearchCV.

Used an 80/20 train-test split for better generalization.

5️⃣ Evaluation

Metrics Used:

Accuracy: Measures overall performance.

Precision & Recall: Ensures balanced evaluation.

Classification Report: Provides detailed insights into model performance.

🚀 Model Improvement Strategies

Try different models: Naive Bayes, Support Vector Machines (SVM).

Increase training data: More labeled data improves accuracy.

Fine-tune hyperparameters: Experiment with different values.

Use Word Embeddings: Instead of TF-IDF, use Word2Vec or BERT for deeper understanding.

🛠 Technologies Used

Python 

NLP Libraries: nltk, re, string

Machine Learning: sklearn

Data Handling: pandas, numpy

📜 How to Run the Project

Clone this repository:

git clone https://github.com/Atharv279/Sentiment-Analysis-Reviews.git

Install required libraries:

pip install numpy pandas sklearn nltk

Download and extract the IMDb dataset.

Open Sentiment_Analysis.ipynb in Jupyter Notebook.

Run all the cells to train and evaluate the model.

📌 Results

The model achieved high accuracy in classifying reviews.

Successfully distinguishes positive and negative sentiments.

📄 Author

👤 Atharv Patil | LinkedIn - www.linkedin.com/in/atharv-patil-bab53a284