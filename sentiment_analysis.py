from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


class SentimentAnalysis:
    def __init__(self, data, content_column='content', date_column='Date'):
        """
        Initializes the SentimentAnalysis class with the given data.
        
        Args:
        data (pd.DataFrame): DataFrame containing the text and date columns.
        content_column (str): Name of the column containing text (default is 'content').
        date_column (str): Name of the column containing dates (default is 'Date').
        """
        self.data = data
        self.content_column = content_column
        self.date_column = date_column
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_score(self, text):
        """
        Gets the sentiment score for a given text using VADER SentimentIntensityAnalyzer.
        
        Args:
        text (str): The text for which sentiment score needs to be calculated.
        
        Returns:
        float: The compound sentiment score.
        """
        sentiment = self.analyzer.polarity_scores(text)
        return sentiment['compound']

    def add_sentiment_scores(self):
        """
        Adds a column to the DataFrame based on the content column.
        """
        self.data['sentiment_score'] = self.data[self.content_column].apply(self.get_sentiment_score)
        return self.data  # Return the updated DataFrame

    def classify_sentiment(self, score):
        """
        Classifies the sentiment as Positive, Negative, or Neutral based on the score.
        
        Args:
        score (float): The sentiment score to classify.
        
        Returns:
        str: 'Positive', 'Negative', or 'Neutral'.
        """
        if score > 0:
            return 'Positive'
        elif score < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def add_overall_sentiment(self):
        """
        Adds a column to the DataFrame based on the Sentiment Score.
        """
        self.data['avg_sentiment'] = self.data['sentiment_score'].apply(self.classify_sentiment)
        return self.data  # Return the updated DataFrame

    def get_date_sentiment(self):
        """
        Groups the data by date and calculates the average sentiment score for each date.
        
        Returns:
        pd.DataFrame
        """
        # Group the data by Date and calculate the average sentiment score
        date_sentiment = self.data.groupby(self.date_column)['sentiment_score'].mean().reset_index()

        # Add a column for overall sentiment (positive/negative/neutral)
        date_sentiment['avg_sentiment'] = date_sentiment['sentiment_score'].apply(self.classify_sentiment)
        
        return date_sentiment  # Return the DataFrame with date sentiment
