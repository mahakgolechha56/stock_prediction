import pandas as pd
from sentiment_analysis import SentimentAnalysis
from data_processing import DataProcessing
from feature_engineering_split import FeatureEngineering
from train_forecast import LightGBMModel
import config



#read the data 
data = pd.read_csv('stock_prediction.csv')

stock_columns = config.STOCK_COLUMNS 
content_columns = config.CONTENT_COLUMNS

stock_data = data[stock_columns]
content_data = data[content_columns]

sentiment_analyzer = SentimentAnalysis(content_data,'content','Date')

# Add sentiment scores to the DataFrame and return the updated DataFrame
data_with_sentiment = sentiment_analyzer.add_sentiment_scores()

# Get the average sentiment score by Date and return the result
date_sentiment = sentiment_analyzer.get_date_sentiment()
#print(date_sentiment)

#remove duplicate enteries in stock data as it had multiple enteries because of news for each day
stock_data =  stock_data.drop_duplicates(subset='Date')

new_data = pd.merge(date_sentiment, stock_data, on='Date', how='left')
data_processing = DataProcessing(new_data)

full_data = data_processing.add_missing_dates()

feature_engineering = FeatureEngineering(full_data)

full_data = feature_engineering.feature_data()

split_time = config.split_date


train_data = full_data.query('Date<=@split_time')

train_process = DataProcessing(train_data)
train_data = train_process.forward_fill()

test_data = full_data.query('Date>@split_time')
test_data = test_data.dropnna()
#print(train_data)
#print(test_data)
cols_to_drop = config.COLS_to_DROP

categorical_cols = config.CATEGORICAL_COLS

X_train = train_data.drop(cols_to_drop,axis = 1)
y_train = train_data['Adj Close']
X_test = test_data.drop(cols_to_drop,axis = 1)
y_test = test_data['Adj Close']


lightgbm_model = LightGBMModel(X_train, y_train, X_test, y_test, categorical_cols)

# Prepare data
lightgbm_model.prepare_data()

# Train the model
lightgbm_model.train_model()

# Make predictions
lightgbm_model.predict()

# Evaluate the model
lightgbm_model.evaluate_model()


# Get predictions and add to a DataFrame
predictions = lightgbm_model.get_predictions()


test_data['predicted'] = predictions
test_data.to_csv('results.csv')







