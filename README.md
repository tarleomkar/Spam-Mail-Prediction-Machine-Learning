# Spam-Mail-Prediction-Machine-Learning
📧 Spam or Ham Mail Classifier
A Logistic Regression based machine learning model to classify emails as spam or ham (legitimate). This project uses TF-IDF vectorization to convert text data into numerical features for model training and testing.

🚀 How It Works
1. Data Preprocessing 🧹
• Load and clean the email dataset (mail_data.csv).
• Replace missing values with empty strings.
• Convert email categories (ham = 1, spam = 0).

2. Text Vectorization ✉️ ➡️ 🔢
• Use TfidfVectorizer to transform text data into numerical feature vectors.

3. Model Training 🏋️‍♂️
• Train a Logistic Regression model using the processed data.

Prediction 🤖
• Use the trained model to predict whether new emails are ham or spam.

🛠 Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

📂 Data Preprocessing
# Load the data
raw_mail_data = pd.read_csv('/content/mail_data.csv')

# Replace null values with empty strings
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Label encoding: spam = 0, ham = 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate features and labels
X = mail_data['Message']
Y = mail_data['Category']

🔄 Splitting the Data
# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

✨ Feature Extraction
# Transform text data into TF-IDF features
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

⚙️ Model Training
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

📊 Model Evaluation
# Accuracy on training data
accuracy_on_training_data = accuracy_score(Y_train, model.predict(X_train_features))
print('Accuracy on training data: ', accuracy_on_training_data)

# Accuracy on test data
accuracy_on_test_data = accuracy_score(Y_test, model.predict(X_test_features))
print('Accuracy on test data: ', accuracy_on_test_data)

🧠 Predictive System
input_mail = ["I've been searching for the right words to thank you..."]

# Convert input to feature vector
input_data_features = feature_extraction.transform(input_mail)

# Make prediction
prediction = model.predict(input_data_features)

# Output the result
if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')

🏆 Results
Training Data Accuracy: 96.77%
Test Data Accuracy: 96.68%
