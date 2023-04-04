import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import ne_chunk

# Load the tweets from the dataset file
tweets_df = pd.read_csv("fifa_world_cup_2022_tweets.csv")

# Define the stopwords
stop_words = set(stopwords.words("english"))

# Define the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Ask the user for the number of tweets to analyze
num_tweets = int(input("Enter the number of tweets to analyze: "))
print("\n")

# Perform the analysis
while True:
    print("Select the type of analysis:")
    print("1. Sentiment Analysis")
    print("2. Keyword Extraction")
    print("3. Emotion Extraction")
    print("4. Stance Extraction")
    print("5. Named Entities Recognition")
    print("6. Exit")

    # Ask the user for the type of analysis
    choice = int(input("Enter your choice: "))

    if choice == 1:
        # Sentiment Analysis
        print("Sentiment analysis on",num_tweets,"tweets")
        for i in range(num_tweets):
            tweet = tweets_df['Tweet'][i]
            sentiment = sia.polarity_scores(tweet)
            label = tweets_df['Sentiment'][i]
            print(i, ">>>", sentiment)

            if sentiment['compound'] >= 0.05:
                predicted_label = 'positive'
            elif sentiment['compound'] <= -0.05:
                predicted_label = 'negative'
            else:
                predicted_label = 'neutral'

            print("Tweet: ", tweet)
            print("Predicted Label: ", predicted_label)
            print("Actual Label: ", label)
        print("\n")

    elif choice == 2:
        # Keyword Extraction
        print("Keyword Extraction on",num_tweets,"tweets")
        for i in range(num_tweets):
            tweet = tweets_df['Tweet'][i]
            # Tokenize the tweet
            tokens = word_tokenize(tweet)
            # Remove stopwords
            filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
            # Find the most common words
            fdist = FreqDist(filtered_tokens)
            print(i, ">>>", fdist.most_common(10))
        print("\n")

    elif choice == 3:
        # Emotion Extraction
        print("Emotion Extraction on",num_tweets,"tweets")
        for i in range(num_tweets):
            tweet = tweets_df['Tweet'][i]
            sentiment = sia.polarity_scores(tweet)
            if sentiment['compound'] >= 0.05:
                print(i, ">>>", "Positive")
            elif sentiment['compound'] <= -0.05:
                print(i, ">>>", "Negative")
            else:
                print(i, ">>>", "Neutral")
        print("\n")

    elif choice == 4:
        # Stance Extraction
        print("Stance Extraction on",num_tweets,"tweets")
        for i in range(num_tweets):
            tweet = tweets_df['Tweet'][i]
            stance = "neutral"  # this would be a model or function call to determine stance
            print(i, ">>>", stance)
        print("\n")

    elif choice == 5:
        # Named Entities Recognition
        print("Named Entities Recognition on",num_tweets,"tweets")
        for i in range(num_tweets):
            tweet = tweets_df['Tweet'][i]
            tokens = nltk.word_tokenize(tweet)
            tagged = nltk.pos_tag(tokens)
            named_entities = nltk.ne_chunk(tagged)
            print(i, ">>>", named_entities)
        print("\n")

    elif choice == 6:
        # Exit
        print("Exiting program...")
        break