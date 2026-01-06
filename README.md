Twitter Sentiment Analysis Using spaCy and Emoji Handling

Project Overview
This project implements a sentiment analysis system to classify Twitter tweets into three sentiment categories: Positive, Negative, and Neutral. The system uses Natural Language Processing (NLP) techniques with spaCy for text preprocessing, incorporates emoji sentiment handling to capture emotional context, and applies a machine learning model for classification.

The project demonstrates a complete NLP pipeline from data preprocessing to model training and prediction, making it suitable for academic projects, internships, and practical machine learning applications.

Objectives
- To preprocess Twitter text data using spaCy NLP pipeline
- To handle emoji-based sentiment for improved accuracy
- To convert text data into numerical features using TF-IDF
- To train a machine learning model for sentiment classification
- To predict sentiment for unseen tweets

Dataset Description
The project uses the Twitter Entity Sentiment Analysis dataset from Kaggle, which is provided in two files:
- twitter_training.csv
- twitter_validation.csv

Each dataset contains the following information:
- Tweet ID
- Entity
- Sentiment label (Positive, Negative, Neutral, Irrelevant)
- Tweet text

For this project, irrelevant samples are removed and only text and sentiment columns are used.

Technologies and Libraries Used
- Python
- spaCy
- scikit-learn
- Pandas
- Emoji library

Methodology

1. Data Loading
Training and validation datasets are loaded separately and then merged into a single dataset for preprocessing and modeling.

2. Data Preprocessing
- Conversion of text to lowercase
- Removal of URLs, mentions, and hashtags
- Removal of stopwords
- Lemmatization using spaCy
- Handling of missing or empty text values

3. Emoji Sentiment Handling
A custom emoji sentiment dictionary is used to assign positive or negative scores to emojis. Based on the cumulative emoji score, a sentiment indicator token is appended to the cleaned text.

4. Feature Extraction
TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert cleaned text into numerical feature vectors suitable for machine learning models.

5. Model Training
A Multinomial Naive Bayes classifier is trained on the TF-IDF features to learn patterns between tweet content and sentiment labels.

6. Model Evaluation
The model is evaluated using accuracy, precision, recall, and F1-score on a held-out test set.

7. Prediction
The trained model can predict sentiment for new, unseen tweets provided by the user.

Results
The model successfully classifies tweets into Positive, Negative, and Neutral categories. Emoji sentiment handling improves performance on emotionally expressive tweets, which are common on social media platforms like Twitter.

How to Run the Project
1. Upload the training and validation CSV files to the working directory
2. Install required dependencies
3. Run the provided Python script or notebook sequentially
4. Use the prediction function to classify new tweets

Sample Prediction
Input:
I love this phone

Output:
Positive

Input:
Worst service ever

Output:
Negative

Applications
- Social media sentiment monitoring
- Brand and product review analysis
- Customer feedback analysis
- Opinion mining

Conclusion
This project demonstrates an end-to-end sentiment analysis system using NLP and machine learning techniques. By combining spaCy-based text preprocessing with emoji sentiment handling, the model achieves more realistic sentiment interpretation for Twitter data. The project is scalable and can be extended to real-time sentiment analysis using Twitter APIs or advanced deep learning models.

Future Enhancements
- Integration with Twitter API for real-time data
- Deployment as a web application using Streamlit
- Use of transformer-based models such as BERT
- Visualization of sentiment trends

Author
Tanushree Rathod
