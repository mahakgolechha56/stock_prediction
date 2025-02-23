
import pandas as pd
import config

class DataProcessing:
    def __init__(self, df):
        """
        Initializes the Reindexer with a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to be reindexed.
        """
        self.df = df
        self.feature_columns = config.FEATURE_COLUMNS 

    def add_missing_dates(self):
        """
        Reindexes the DataFrame based on the date range from the min to the max date,
        filling missing dates 

        Returns:
            pandas.DataFrame: The reindexed DataFrame with filled missing dates.
        """
        # Ensure the 'date' column is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Create a full date range from the min to the max date in the 'date' column
        full_date_range = pd.date_range(start=self.df['Date'].min(), end=self.df['Date'].max(), freq='D')

        # Reindex the DataFrame to the full date range, filling missing dates as NaN
        df_full = self.df.set_index('Date').reindex(full_date_range).reset_index()

        # Rename the 'index' column to 'date'
        df_full.rename(columns={'index': 'Date'}, inplace=True)

        return df_full

    def forward_fill(self):
        """
        Forward fill the missing values 
        """
        self.df[self.feature_columns] = self.df[self.feature_columns].fillna(method='ffill')
        return self.df






