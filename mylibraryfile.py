#!/usr/bin/env python
# coding: utf-8

# # CAI 2820C - AI Applications Solutions
# 
# ## Spring 2025 - Laura Castillo

# ## Setting up the environment




# In[3]:


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # In case the attribute is missing (older versions of Python)
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Now download stopwords
nltk.download('stopwords')


# In[4]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import joblib
import os


# ## Collecting Data

# In[5]:


data = pd.read_excel("AllITBooks_DataSet.xlsx")
data.head()


# In[6]:


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# In[7]:


data["ConsolidatedText"] = data.Book_name + " " + data.Sub_title + " " + data.Description


# In[8]:


data["ConsolidatedText"]


# In[9]:


stop_words = set(stopwords.words('english'))
list(stop_words)[:5]


# ## Cleaning text

# In[10]:


def preprocess_text(text):

    text = str(text)

    text = text.lower()

    tokens = word_tokenize(text)

    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    return " ".join(filtered_tokens)


# In[11]:


data["CleanedDescription"] = data["ConsolidatedText"].apply(preprocess_text)


# In[12]:


data[["ConsolidatedText","CleanedDescription"]]


# ## Creating a word embedding

# In[13]:


tfidf_vectorizer = TfidfVectorizer(max_features=500)

tfidf_matrix = tfidf_vectorizer.fit_transform(data["CleanedDescription"])


# In[14]:


tfidf_vectorizer.get_feature_names_out()


# In[15]:


tfidfmatrix = pd.DataFrame(tfidf_matrix.toarray()  , columns=tfidf_vectorizer.get_feature_names_out())


# In[16]:


tfidfmatrix.head()


# ## Modeling

# In[17]:


# K-Means for Topic Modeling
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42)

kmeans.fit(tfidf_matrix)


# ## Get topics

# In[18]:


feature_names = tfidf_vectorizer.get_feature_names_out()

topics = []

for topic_idx, topic in enumerate(kmeans.cluster_centers_):

    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]

    print(top_words)

    topics.append(", ".join(top_words))


# In[19]:


categories = {
    0: "Content Management Systems (CMS)",
    1: "Web Development and Frameworks",
    2: "Data Analysis and Big Data",
    3: "Game Development",
    4: "Network and Security Administration",
    5: "Programming Languages and Functional Programming",
    6: "Mobile App Development",
    7: "Java and Enterprise Applications",
    8: "Python and Machine Learning",
    9: "Databases and SQL Administration"
}


# In[20]:


topic_assignments = kmeans.predict(tfidf_matrix)


# In[21]:


topic_assignments


# In[22]:


data["AssignedTopic"] = topic_assignments
data['Topic_Keywords'] = [topics[i] for i in topic_assignments]
data['Topic'] = data['AssignedTopic'].map(categories)


# In[23]:


data


# In[24]:


print(data.Description[0])


# In[25]:


print(data.Description[1])


# Predicting a new category based on a book description

# In[26]:


bookdescription = """As data floods into your company, you need to put it to work right away―and SQL is the best tool for the job. With the latest edition of this introductory guide, author Alan Beaulieu helps developers get up to speed with SQL fundamentals for writing database applications, performing administrative tasks, and generating reports. You’ll find new chapters on SQL and big data, analytic functions, and working with very large databases.

Each chapter presents a self-contained lesson on a key SQL concept or technique using numerous illustrations and annotated examples. Exercises let you practice the skills you learn. Knowledge of SQL is a must for interacting with data. With Learning SQL, you’ll quickly discover how to put the power and flexibility of this language to work.

Move quickly through SQL basics and several advanced features
Use SQL data statements to generate, manipulate, and retrieve data
Create database objects, such as tables, indexes, and constraints with SQL schema statements
Learn how datasets interact with queries; understand the importance of subqueries
Convert and manipulate data with SQL’s built-in functions and use conditional logic in data statements"""


# In[27]:


# clean the data
bookdescription_cleaned = preprocess_text(bookdescription)

#vectorize the text
bookdecription_vectorized = tfidf_vectorizer.transform([bookdescription_cleaned])

#getting topics
topic_result = kmeans.predict(bookdecription_vectorized)
topic_result


#getting the top topic
book_category = categories.get(int(topic_result))

book_category


# In[28]:


pd.DataFrame(bookdecription_vectorized.toarray(), columns= tfidf_vectorizer.get_feature_names_out())


# In[29]:


if not os.path.exists("models"):
    os.mkdir("models")

joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")


# Working with ipywidgets

# In[30]:


def categorizeBooks(bookdescription):

    # clean the data
    bookdescription_cleaned = preprocess_text(bookdescription)

    #vectorize the text
    bookdecription_vectorized = tfidf_vectorizer.transform([bookdescription])

    #getting topics
    topic_result = kmeans.predict(bookdecription_vectorized)

    #getting the top topic
    book_category = categories.get(int(topic_result))

    if topic_result.max()<=0.01:
        book_category = "Unknown"

    return book_category



# In[32]:


import ipywidgets as widgets
from ipywidgets import interactive, interact
from IPython.display import display


# In[33]:


@interact(book_description=widgets.Textarea(
    value=None,
    placeholder="Type a book description...",
    description="Book Description:",
    layout=widgets.Layout(width="500px", height="150px")
))
def interactive_categorize(book_description):
    category = categorizeBooks(book_description)
    print(category)


# Extra Credit

# In[34]:


from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between all topics
topic_similarities = cosine_similarity(kmeans.cluster_centers_)

# Function to get the two most related topics
def get_related_topics(topic_index):
    similarities = topic_similarities[topic_index]
    # Get indices of the two most similar topics (excluding the topic itself)
    related_topic_indices = similarities.argsort()[-3:-1]
    # Get the names of the related topics
    related_topics = [categories[i] for i in related_topic_indices]
    return related_topics


# In[35]:


# Apply the function to your DataFrame
data['Related_Topics'] = data['AssignedTopic'].apply(get_related_topics)

# Display the main category and related topics
for index, row in data.iterrows():
    print(f"Book: {row['Book_name']}")
    print(f"Main Category: {row['Topic']}")
    print(f"Related Topics: {row['Related_Topics']}")
    print("-" * 20)


# 

# In[37]:


import streamlit as st


# In[38]:


# Streamlit UI
import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained models
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
kmeans = joblib.load("models/kmeans_model.pkl")

# Categories definition
categories = {
    0: "Content Management Systems (CMS)",
    1: "Web Development and Frameworks",
    2: "Data Analysis and Big Data",
    3: "Game Development",
    4: "Network and Security Administration",
    5: "Programming Languages and Functional Programming",
    6: "Mobile App Development",
    7: "Java and Enterprise Applications",
    8: "Python and Machine Learning",
    9: "Databases and SQL Administration"
}

# Function to clean and categorize books
def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    
    text = str(text).lower()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

def categorizeBooks(bookdescription):
    bookdescription_cleaned = preprocess_text(bookdescription)
    bookdecription_vectorized = tfidf_vectorizer.transform([bookdescription_cleaned])
    topic_result = kmeans.predict(bookdecription_vectorized)
    book_category = categories.get(int(topic_result[0]), "Unknown")
    return topic_result[0], book_category

# Function to get related topics using cosine similarity
def get_related_topics(topic_index):
    topic_similarities = cosine_similarity(kmeans.cluster_centers_)
    similarities = topic_similarities[topic_index]
    related_topic_indices = similarities.argsort()[-3:-1]  # Get top 2 related topics
    related_topics = [categories[i] for i in related_topic_indices]
    return related_topics

# Streamlit UI
st.title("📚 AI Book Categorizer with Related Topics")
st.write("This application automatically categorizes books based on their description and suggests related topics.")
st.markdown("---")

# User input section
st.subheader("🔍 Enter Book Description:")
bookdescription = st.text_area("Book Description:", placeholder="Type or paste the book description here...", height=150)

if st.button("📖 Categorize Book"):
    if len(bookdescription.strip()) == 0:
        st.error("⚠️ Please enter a valid book description!")
    else:
        # Categorize book and fetch related topics
        topic_index, category = categorizeBooks(bookdescription)
        related_topics = get_related_topics(topic_index)
        
        # Display the main category and related topics
        st.success(f"**Main Category:** {category}")
        st.write("### 🔗 **Related Topics:**")
        for topic in related_topics:
            st.markdown(f"- {topic}")

        st.markdown("---")
        st.info("This categorization is powered by machine learning models trained using KMeans clustering and TF-IDF word embeddings.")
