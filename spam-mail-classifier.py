#!/usr/bin/env python
# coding: utf-8

# <div style="
#     background: linear-gradient(to right, #f0f0f0, #e6e6e6); 
#     padding: 30px; 
#     border-radius: 15px;
#     color: Brown; /* Vibrant orange */
#     font-weight: bold; 
#     text-align: center;
#     font-size: 28px; 
#     font-family: 'Verdana', sans-serif;
#     box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
# ">
#     Spam SMS Classifier
# </div>

# ![image.png](attachment:image.png)

# <div style="
#     padding: 15px; 
#     color: brown; /* Darker orange */
#     font-weight: bold; 
#     font-size: 20px; 
#     font-family: 'Verdana', sans-serif;
#     text-align: left; /* Left-aligned */
#     box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
#     margin-top: 10px; /* Space between title and subtitle */
# ">
#     Import Libraries:
# </div>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize

from collections import Counter


# In[4]:


import warnings
warnings .filterwarnings("ignore")


# In[5]:


pd.set_option("display.max_columns",None)
pd.set_option('display.max_rows',None)


# <div style="
#     padding: 15px; 
#     color: brown; /* Darker orange */
#     font-weight: bold; 
#     font-size: 20px; 
#     font-family: 'Verdana', sans-serif;
#     text-align: left; /* Left-aligned */
#     box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
#     margin-top: 10px; /* Space between title and subtitle */
# ">
#     Load Data:
# </div>
# 

# In[6]:


df=pd.read_csv('C:/Users/Effat/Desktop/Internships/CODSOFT/spam.csv',encoding_errors= 'replace')
df.head()


# <div style="
#     padding: 15px; 
#     color: brown; /* Darker orange */
#     font-weight: bold; 
#     font-size: 20px; 
#     font-family: 'Verdana', sans-serif;
#     text-align: left; /* Left-aligned */
#     box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
#     margin-top: 10px; /* Space between title and subtitle */
# ">
#     Exploratory Data Analytics - EDA
# </div>
# 

# In[7]:


# i will drop these colomns they aren't useful
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)


# In[8]:


# Rename columns for better clarity 
df.columns = ['Label', 'Message']


# In[9]:


df.head()


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


df.isna().sum()


# In[13]:


df.describe(include='object')


# In[14]:


df.duplicated().sum()


# In[15]:


# Count duplicated rows
duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")

# Prepare data
unique_count = len(df) - duplicate_count
row_types = ['Unique Rows', 'Duplicate Rows']
counts = [unique_count, duplicate_count]

plt.figure(figsize=(8, 6))
plt.pie(
    counts, 
    labels=row_types, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=["#FFA07A", "#D3D3D3"], 
    explode=(0.05, 0.15),  
    textprops={'fontsize': 14, 'color': '#000000', 'fontweight': 'bold'}  
)
plt.title('Duplicate Rows Distribution', fontsize=18, color='#000000', fontweight='bold')  
plt.show()


# In[16]:


# Show duplicate rows
duplicates = df[df.duplicated()]
print(duplicates.head())


# In[17]:


# Check the label distribution among duplicates
print(duplicates['Label'].value_counts())


# In[18]:


df=df.drop_duplicates()


# In[19]:


df.shape


# In[20]:


df.duplicated().sum()


# In[21]:


# Count the number of ham and spam messages
label_counts = df['Label'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.pie(
    label_counts, 
    labels=['Ham', 'Spam'],  
    autopct='%1.1f%%', 
    startangle=140, 
    colors=["brown", "#D3D3D3"],  
    explode=(0.05, 0.15),  
    textprops={'fontsize': 14, 'color': '#000000', 'fontweight': 'bold'}  
)
plt.title('Proportion of Ham vs Spam', fontsize=18, color='#000000', fontweight='bold')  
plt.show()



# In[27]:


from wordcloud import WordCloud

# Word Cloud for Ham
ham_words = ' '.join(df[df['Label'] == 'ham']['Message'])
ham_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='RdPu').generate(ham_words)

# Word Cloud for Spam
spam_words = ' '.join(df[df['Label'] == 'spam']['Message'])
spam_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='BrBG').generate(spam_words)

# Display word clouds
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(ham_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Ham Messages', fontsize=16, color='purple', fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Spam Messages', fontsize=16, color='Brown', fontweight='bold')
plt.axis('off')

plt.show()


# In[ ]:


spam_corpus = []
for msg in df[df['Message'] == 1]['transform_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[28]:


# Adding Word Count Feature
df['Word_Count'] = df['Message'].apply(lambda x: len(str(x).split()))

# Adding Character Count Feature
df['Char_Count'] = df['Message'].apply(lambda x: len(str(x)))

# Adding Average Word Length Feature
# Handling division by zero to avoid errors for messages with zero words
df['Avg_Word_Length'] = df.apply(
    lambda row: row['Char_Count'] / row['Word_Count'] if row['Word_Count'] > 0 else 0, axis=1
)


# In[29]:


df.head()


# In[30]:


df.describe()


# <div style="
#     padding: 15px; 
#     color: brown; /* Darker orange */
#     font-weight: bold; 
#     font-size: 20px; 
#     font-family: 'Verdana', sans-serif;
#     text-align: left; /* Left-aligned */
#     box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
#     margin-top: 10px; /* Space between title and subtitle */
# ">
#     Preprocess Data:
# </div>

# In[31]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Word_Count', 'Char_Count', 'Avg_Word_Length']] = scaler.fit_transform(df[['Word_Count', 'Char_Count', 'Avg_Word_Length']])


# In[32]:


def normalize_text(text):
    """
    Enhanced text normalization: lowercasing, punctuation removal, stopword removal,
    abbreviation handling, and rare word removal.
    """
    # Initialize stopwords
    stop_words = set(stopwords.words('english'))
    
    # Custom abbreviations dictionary
    abbreviations = {
        "la": "",
        "wat": "what",
        "n": "and",
        "ur": "your",
        "u": "you"
    }
    
    # Convert text to lowercase
    text = text.lower()
    
    # Expand abbreviations
    words = text.split()
    words = [abbreviations.get(word, word) for word in words]
    text = " ".join(words)
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove punctuation, non-alphabetic characters, and stopwords
    words = [re.sub(r"[^\w\s]", "", word) for word in words if word.isalpha() and word not in stop_words]
    
    # Rare words removal (can be applied after initial processing)
    word_counts = Counter(" ".join(df['Message']).split())
    rare_words = {word for word, count in word_counts.items() if count < 2}
    words = [word for word in words if word not in rare_words]
    
    # Join words back into a single string
    normalized_text = " ".join(words)
    
    return normalized_text

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional, to ensure full support for lemmatization

# In[33]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

# Initialize Stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Custom abbreviation mapping
abbreviations = {
    "u": "you",
    "ur": "your",
    "la": "the",  # Optional: Add abbreviation mappings here
    "wat": "what",
    "n": "and",
    "e": "the"
}

# Function to handle text normalization with stemming, stopwords, and abbreviation handling
def normalize_text(text):
    """
    Simplified text normalization: lowercasing, punctuation removal, stopword removal, 
    abbreviation handling, and stemming.
    """
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Expand abbreviations
    words = [abbreviations.get(word, word) for word in words]

    # Remove punctuation, stopwords, and non-alphabetic words
    words = [re.sub(r"[^\w\s]", "", word) for word in words if word.isalpha() and word not in stop_words]

    # Stem words
    words = [stemmer.stem(word) for word in words]

    # Join words back into a single string
    normalized_text = " ".join(words)
    return normalized_text


# In[34]:


# Apply the normalize_text function to the 'Message' column
df['Normalized_Message'] = df['Message'].apply(normalize_text)


# In[35]:


df.head()


# In[36]:


# Set Pandas options to display the full content of columns without truncation
pd.set_option('display.max_colwidth', None)

# Display the first row's full original and normalized message
row = df.iloc[0]
print(f"Message: {row['Message']}")
print(f"Normalized_Message: {row['Normalized_Message']}")



# In[37]:


from collections import Counter

spam_words = Counter(" ".join(df[df['Label'] == 'spam']['Normalized_Message']).split())
ham_words = Counter(" ".join(df[df['Label'] == 'ham']['Normalized_Message']).split())

# Print spam words
print("Most Common Words in Spam Messages:")
print("-" * 35)
print(f"{'Word':<10} | {'Count':<10}")
print("-" * 35)
for word, count in spam_words.most_common(10):
    print(f"{word:<10} | {count:<10}")
print("\n")

# Print ham words
print("Most Common Words in Ham Messages:")
print("-" * 35)
print(f"{'Word':<10} | {'Count':<10}")
print("-" * 35)
for word, count in ham_words.most_common(10):
    print(f"{word:<10} | {count:<10}")



# In[38]:


# Encode labels
df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})


# In[39]:


from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# Split features and labels
X = df['Message']
y = df['Label']

# Convert text data into numerical form using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_transformed, y)

print("Before SMOTE:", y.value_counts())
print("After SMOTE:", pd.Series(y_resampled).value_counts())


# <div style="
#     padding: 15px; 
#     color: brown; /* Darker orange */
#     font-weight: bold; 
#     font-size: 20px; 
#     font-family: 'Verdana', sans-serif;
#     text-align: left; /* Left-aligned */
#     box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
#     margin-top: 10px; /* Space between title and subtitle */
# ">
#     Modeling:
# </div>

# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("Training set class distribution:")
print(pd.Series(y_train).value_counts())


# In[41]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


model = MultinomialNB()
model.fit(X_train, y_train)


# In[42]:


# Predictions
y_pred = model.predict(X_test)


# In[43]:


# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[44]:


print("Classification Report:\n", classification_report(y_test, y_pred))


# <div style="
#     padding: 15px; 
#     color: brown; /* Darker orange */
#     font-weight: bold; 
#     font-size: 20px; 
#     font-family: 'Verdana', sans-serif;
#     text-align: left; /* Left-aligned */
#     box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
#     margin-top: 10px; /* Space between title and subtitle */
# ">
#     Save Model:
# </div>

# In[45]:


import joblib
churn_model=model
# Save the model
joblib.dump(churn_model, "model.pkl")


# In[46]:


joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("TF-IDF vectorizer loaded successfully.")


# <div style="
#     padding: 15px; 
#     color:brown; /* Darker orange */
#     font-weight: bold; 
#     font-size: 20px; 
#     font-family: 'Verdana', sans-serif;
#     text-align: left; /* Left-aligned */
#     box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
#     margin-top: 10px; /* Space between title and subtitle */
# ">
#     Predict New Data:
# </div>

# In[47]:


def predict_message(message, model, vectorizer):
    # Define label mapping
    label_mapping = {
        0: "Not Spam", 
        1: "Spam"       
    }
    
    # Preprocess the message
    message_transformed = vectorizer.transform([message])
    
    # Predict the class and probabilities
    predicted_class = model.predict(message_transformed)[0]
    predicted_probabilities = model.predict_proba(message_transformed)[0]
    
    # Map the predicted class to its label
    predicted_label = label_mapping[predicted_class]
    
    return predicted_label, predicted_probabilities



# In[50]:


new_message ="Hello Iffat, I hope everything is going smoothly on your end. Just a quick reminder about our call scheduled for 2 PM tomorrow. Please let me know if there are any changes to your availability. Looking forward to connecting!"
predicted_class, predicted_probabilities = predict_message(new_message, model, vectorizer)
print(f"Predicted Class: {predicted_class}")
print(f"Prediction Probabilities {predicted_probabilities}")


# In[53]:


new_message = "Hey,you get credit of 30000"
predicted_class, predicted_probabilities = predict_message(new_message, model, vectorizer)
print(f"Predicted Class: {predicted_class}")
print(f"Prediction Probabilities: {predicted_probabilities}")

