STOCK_COLUMNS = ['ticker','Date','Open','High','Low','Close','Adj Close','Volume']
CONTENT_COLUMNS = ['Date','content']
FEATURE_COLUMNS = ['avg_sentiment','ticker','Open','High','Low','Close','Adj Close','Volume']	
split_date = "2018-12-31"

COLS_to_DROP = ['sentiment_score','ticker','Close','Adj Close','Date','month']

CATEGORICAL_COLS = ['avg_sentiment','is_quarter_end','day']
