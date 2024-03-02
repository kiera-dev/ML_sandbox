import os
import string
import nltk
print(nltk.data.path)
print('dfawfaawfafawfawawfawf')
#if punkt is erroring  on ssl: bash '/Applications/Python 3.11/Install Certificates.command'
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer



def read_emails(folder):
    emails = []
    labels = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            emails.append(content)
            labels.append(folder)  # Use folder name as label (spam or ham)
    return emails, labels

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text



spam_folder = "enron1/spam"
ham_folder = "enron1/ham"

# Read emails
spam_emails, spam_labels = read_emails(spam_folder)
ham_emails, ham_labels = read_emails(ham_folder)

# Combine spam and ham emails
all_emails = spam_emails + ham_emails
all_labels = spam_labels + ham_labels

preprocessed_spam_emails = []
for email in spam_emails:
    preprocessed_email = preprocess_text(email)
    preprocessed_spam_emails.append(preprocessed_email)


preprocessed_ham_emails = []
for email in ham_emails:
    preprocessed_email = preprocess_text(email)
    preprocessed_ham_emails.append(preprocessed_email)


preprocessed_all_emails = []
for email in all_emails:
    preprocessed_email = preprocess_text(email)
    preprocessed_all_emails.append(preprocessed_email)


print("Number of spam emails:", len(spam_emails))
print("Number of ham emails:", len(ham_emails))
print("Total number of emails:", len(all_emails))

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_all_emails)
print("Shape of TF-IDF matrix:", tfidf_matrix.shape)

