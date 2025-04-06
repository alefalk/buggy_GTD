import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer





class RF_predictions:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path, encoding='ISO-8859-1')
        self.data['attack_date'] = pd.to_datetime({'year': self.data['iyear'], 'month': self.data['imonth'], 'day': self.data['iday']})

        self.data.sort_values(by=['gname', 'attack_date'], inplace=True)
        self.data.drop(columns=['attack_date'])

        self.train, self.test = self.handle_leakage()

    def splitting(self, train_df, test_df):
        y_train = train_df['gname']
        X_train = train_df.drop(columns=['gname'])
        y_test = test_df['gname']
        X_test = test_df.drop(columns=['gname'])
        return X_train, X_test, y_train, y_test
    
    def handle_leakage(self):
        train_frames = []
        test_frames = []

        #first 70% of each groups attacks to training set, remainin 30% to testing set
        for _, group_data in self.data.groupby('gname'):
            split_point = int(len(group_data) * 0.7)  # 70% for training
            train_frames.append(group_data.iloc[:split_point])
            test_frames.append(group_data.iloc[split_point:])           


        # Concatenate all the group-specific splits into final train and test DataFrames
        train_df = pd.concat(train_frames)
        test_df = pd.concat(test_frames)


        #drop temporal data
        train_df = train_df.drop(columns=['iyear', 'imonth', 'iday', 'attack_date'])
        test_df = test_df.drop(columns=['iyear', 'imonth', 'iday', 'attack_date'])

        # Shuffle each DataFrame separately
        train_df = shuffle(train_df)
        test_df = shuffle(test_df)

        print(len(train_df))

        return train_df, test_df
    
    # Find optimal parameters for data
    def randomizedSearch(self, X_train, y_train):
        param_grid_rfc = {
            'criterion': ["gini", "entropy"],
            'n_estimators': [5, 10, 20, 50, 100, 150, 200, 300, 500],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'max_features': ['sqrt', 'log2']
            }

        rfc = RandomForestClassifier(random_state=42)

        tscv = TimeSeriesSplit(n_splits=5)

        rs_rfc = RandomizedSearchCV(
            estimator=rfc,
            param_distributions=param_grid_rfc,
            scoring='f1_weighted',
            refit=True,
            n_iter=10,
            return_train_score=True,
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )

        gb_train = rs_rfc.fit(X_train, y_train)
        best_gb = rs_rfc.best_estimator_
        best_params = rs_rfc.best_params_
        best_gb_index = rs_rfc.best_index_
        return best_params
    
    def train_best_params_rf(self, best_params, X_train, y_train):
        rfc = RandomForestClassifier(**best_params, random_state=42)    
        rfc.fit(X_train, y_train)
        return rfc

    def make_predictions(self, best_rfc, X_test, y_test):
        y_pred_gbc = best_rfc.predict(X_test)
        accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
        print(f"Accuracy: {accuracy_gbc * 100:.2f}%")
        return accuracy_gbc, y_pred_gbc




def main(path):
    """Main function to initialize and process data."""
    model = RF_predictions(path)

    X_train, X_test, y_train, y_test = model.splitting(model.train, model.test)

    print("Finding optimal hyperparameters...")
    best_params = model.randomizedSearch(X_train, y_train)

    print("Training best RF classifier...")
    best_rfc = model.train_best_params_rf(best_params, X_train, y_train)

    print("Making predictions...")
    accuracy_gbc, y_pred_gbc = model.make_predictions(best_rfc, X_test, y_test)

    return model, accuracy_gbc, y_pred_gbc, y_test  

if __name__ == "__main__":
    main()  
