import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK stopwords once (uncomment if needed)
# nltk.download('stopwords')

def load_data():
    # Load the datasets from the data folder
    apps_df = pd.read_csv('data/googleplaystore.csv')
    reviews_df = pd.read_csv('data/googleplaystore_user_reviews.csv')
    return apps_df, reviews_df

def filter_health_apps(apps):
    # Select apps in the Health & Fitness category
    health_apps = apps[apps['Category'] == 'HEALTH_AND_FITNESS']
    return health_apps

def filter_five_star_reviews(reviews, app_names):
    # Keep reviews only for apps in health category and with positive sentiment (5-star equivalent)
    reviews_filtered = reviews[reviews['App'].isin(app_names)]
    five_star_reviews = reviews_filtered[reviews_filtered['Sentiment'] == 'Positive']
    return five_star_reviews

def clean_text(text, apps_list):
    # Remove app names and punctuation, convert to lowercase
    for app in apps_list:
        text = text.replace(app, '')  # Remove app names
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    return text

def generate_wordcloud(text):
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    cleaned_text = ' '.join(words)

    # Generate word cloud image
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

    # Plot the word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud from 5-Star Reviews (Health & Fitness Apps)', fontsize=16)
    plt.show()

def main():
    apps, reviews = load_data()
    health_apps = filter_health_apps(apps)
    health_app_names = health_apps['App'].unique()
    
    five_star_reviews = filter_five_star_reviews(reviews, health_app_names)

    all_reviews_text = " ".join(five_star_reviews['Translated_Review'].dropna().tolist())

    cleaned_text = clean_text(all_reviews_text, health_app_names)

    generate_wordcloud(cleaned_text)

if __name__ == "__main__":
    main()
    plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')


