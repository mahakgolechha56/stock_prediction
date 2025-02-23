
import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, data):
        """
        Initializes the DataProcessor with the DataFrame.

        Args:
            data (pandas.DataFrame): The DataFrame to be processed.
        """
        self.data = data

    def feature_data(self):
        """
        Process the DataFrame by adding month, quarter end, and converting columns to categorical.

        Returns:
            pandas.DataFrame: The processed DataFrame with additional columns.
        """
        # Add a 'month' column from the 'Date' column
        self.data['month'] = self.data['Date'].dt.month
        self.data['day'] = self.data['Date'].dt.dayofweek
        # Add 'is_quarter_end' column (1 if quarter end, 0 otherwise)
        self.data['is_quarter_end'] = np.where(self.data['month'] % 3 == 0, 1, 0)

        # Convert 'Overall Sentiment' column to categorical type
        self.data['avg_sentiment'] = self.data['avg_sentiment'].astype('category')
        
        self.data['day'] = self.data['day'].astype('category')
        # Convert 'is_quarter_end' column to categorical type
        self.data['is_quarter_end'] = self.data['is_quarter_end'].astype('category')

        return self.data
