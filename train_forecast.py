import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

class LightGBMModel:
    def __init__(self, X_train, y_train, X_test, y_test, categorical_cols):
        """
        Initializes the LightGBM model class with training and testing data.

        Args:
            X_train (pandas.DataFrame): Features for training
            y_train (pandas.Series): Target variable for training
            X_test (pandas.DataFrame): Features for testing
            y_test (pandas.Series): Target variable for testing
            categorical_cols (list): List of column names that are categorical
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.categorical_cols = categorical_cols
        
        self.model = None
        self.y_pred = None
        self.mse = None

    def prepare_data(self):
        """
        Converts the training and testing data into LightGBM dataset format.
        """
        self.train_data = lgb.Dataset(self.X_train, label=self.y_train, categorical_feature=self.categorical_cols)
        self.test_data = lgb.Dataset(self.X_test, label=self.y_test, categorical_feature=self.categorical_cols, reference=self.train_data)

    def train_model(self):
        """
        Trains the LightGBM model using the provided training data.
        """
        # Define LightGBM parameters
        self.lgbm_params = {
            'objective': 'tweedie',       # Use the Tweedie loss function
            'tweedie_variance_power': 1.5,  # Choose an appropriate power value
            'metric': 'mse',               # Mean squared error (for evaluation)
            'boosting_type': 'gbdt',       # Gradient Boosting Decision Tree
            'num_leaves': 31,              # Max number of leaves per tree
            'learning_rate': 0.05,         # Learning rate
            'feature_fraction': 0.9,       # Fraction of features to use for each boosting round
            'bagging_fraction': 0.8,       # Fraction of data to use for each boosting round
            'bagging_freq': 5,             # Frequency of bagging (subsampling)
            'verbosity': -1                # Suppress detailed logging
        }
        
        # Train the model
        self.model = lgb.train(self.lgbm_params, self.train_data, valid_sets=[self.test_data])

    def feature_importance(self):
        
        feature_importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.X_train.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        return importance_df


    def predict(self):
        """
        Makes predictions on the test data using the trained model.
        """
        self.y_pred = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)

    def evaluate_model(self):
        """
        Evaluates the model performance using Mean Squared Error (MSE).
        """
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        print(f'Mean Squared Error: {self.mse}')

        

    def get_predictions(self):
        """
        Returns the predictions made by the model.
        """
        return self.y_pred


