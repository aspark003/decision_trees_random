import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import os
os.makedirs('../plots', exist_ok=True)

class Regressor:
    def __init__(self, file):
        df = pd.read_pickle(file)
        self.copy = df.copy()
        self.x = self.copy.drop(columns=['median_house_value','total_bedrooms','ocean_proximity'])
        self.y = self.copy['median_house_value']

        self.num = self.x.select_dtypes(include=['number']).columns
        self.obj = self.x.select_dtypes(include=['object','string', 'category']).columns

        self.ss = StandardScaler()
        self.ohe = OneHotEncoder(drop='first', handle_unknown='ignore',sparse_output=False)

        self.n_simple = Pipeline([('n_simple',SimpleImputer(strategy='median', add_indicator=True)),
                                  ('n', self.ss)])

        self.o_simple = Pipeline([('o_simple',SimpleImputer(strategy='constant', fill_value='missing')),
                                  ('o', self.ohe)])

        self.preprocessor = ColumnTransformer([('scaler', self.n_simple, self.num),
                                               ('encoder', self.o_simple, self.obj)])

    def decision(self):
        dtr = DecisionTreeRegressor(random_state=42)

        pipe = Pipeline([('preprocessor', self.preprocessor),
                         ('dtr', dtr)])

        param_grid = {
            'dtr__criterion': ['squared_error'],
            'dtr__max_depth': [30],
            'dtr__min_samples_split': [30],
            'dtr__min_samples_leaf': [15],
            'dtr__max_features': [None],
            'dtr__ccp_alpha': [0.0]
        }

        grid = GridSearchCV(estimator=pipe,
                            param_grid=param_grid,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        grid.fit(X_train, y_train)
        pred = grid.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        residual = y_test - pred

        score = (f'decision tree regressor:\nr2: {r2}\nmae: {mae}\nmse: {mse}\nrmse: {rmse}\n')

        decision_df = pd.DataFrame({'actual': y_test,
                                    'predict': pred,
                                    'residual': residual}).reset_index(drop=True)

        plt.figure(figsize=(10, 8))
        plt.plot(decision_df['actual'], decision_df['actual'], c='green')
        plt.scatter(decision_df['actual'], decision_df['predict'], c='red')
        plt.title('DECISION TREE REGRESSOR\nACTUAL VS PREDICT')
        plt.ylabel('PREDICT')
        plt.xlabel('ACTUAL')
        plt.savefig('../plots/decision_actual_predict.png', dpi=250, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.title('DECISION TREE REGRESSOR\nPREDICT VS RESIDUAL')
        plt.plot(decision_df['predict'], decision_df['predict'], c='green')
        plt.scatter(decision_df['predict'], decision_df['residual'], c='red')
        plt.ylabel('RESIDUAL')
        plt.xlabel('PREDICT')
        plt.savefig('../plots/decision_predict_residual.png', dpi=250, bbox_inches='tight')
        plt.show()

        estimator = grid.best_estimator_

        feature_importance = estimator.named_steps['dtr'].feature_importances_

        importance_df = estimator.named_steps['preprocessor'].get_feature_names_out()

        feature_df = pd.DataFrame({'feature': importance_df,
                                   'importance': feature_importance})

        feature_df = feature_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

        plt.figure(figsize=(10, 8))
        plt.title('DECISION TREE REGRESSOR\nFEATURE IMPORTANCE')
        plt.barh(feature_df['feature'].head(5), feature_df['importance'].head(5))
        plt.ylabel('FEATURE')
        plt.xlabel('RANGE')
        plt.gca().invert_yaxis()
        plt.savefig('../plots/decision_feature_importance.png', dpi=250, bbox_inches='tight')
        plt.show()

        return score, decision_df, feature_df

    def extra(self):
        etr = ExtraTreesRegressor(random_state=42)

        pipe = Pipeline([('preprocessor', self.preprocessor),
                         ('etr', etr)])

        param_grid = {
            "etr__n_estimators": [300],
            "etr__max_depth": [None],
            "etr__min_samples_split": [2],
            "etr__min_samples_leaf": [1],
            "etr__max_features": ["sqrt"]
        }

        grid = GridSearchCV(estimator=pipe,
                            param_grid=param_grid,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        grid.fit(X_train, y_train)
        pred = grid.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        residual = y_test - pred

        score = (f'extra trees regressor:\nr2: {r2}\nmae: {mae}\nmse: {mse}\nrmse: {rmse}\n')

        decision_df = pd.DataFrame({'actual': y_test,
                                    'predict': pred,
                                    'residual': residual}).reset_index(drop=True)

        plt.figure(figsize=(10, 8))
        plt.plot(decision_df['actual'], decision_df['actual'], c='green')
        plt.scatter(decision_df['actual'], decision_df['predict'], c='red')
        plt.title('EXTRA TREES REGRESSOR\nACTUAL VS PREDICT')
        plt.ylabel('PREDICT')
        plt.xlabel('ACTUAL')
        plt.savefig('../plots/extra_actual_predict.png',dpi=250,bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.title('EXTRA TREES REGRESSOR\nPREDICT VS RESIDUAL')
        plt.plot(decision_df['predict'], decision_df['predict'], c='green')
        plt.scatter(decision_df['predict'], decision_df['residual'], c='red')
        plt.ylabel('RESIDUAL')
        plt.xlabel('PREDICT')
        plt.savefig('../plots/extra_predict_residual.png', dpi=250, bbox_inches='tight')
        plt.show()

        estimator = grid.best_estimator_

        feature_importance = estimator.named_steps['etr'].feature_importances_

        importance_df = estimator.named_steps['preprocessor'].get_feature_names_out()

        feature_df = pd.DataFrame({'feature': importance_df,
                                   'importance': feature_importance})

        feature_df = feature_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

        plt.figure(figsize=(10, 8))
        plt.title('EXTRA TREES REGRESSOR\nFEATURE IMPORTANCE')
        plt.barh(feature_df['feature'].head(5), feature_df['importance'].head(5))
        plt.ylabel('FEATURE')
        plt.xlabel('RANGE')
        plt.gca().invert_yaxis()
        plt.savefig('../plots/extra_feature_importance.png', dpi=250, bbox_inches='tight')
        plt.show()
        return score, decision_df, feature_df

    def random(self):
        rfr = RandomForestRegressor(random_state=42)

        pipe = Pipeline([('preprocessor', self.preprocessor),
                         ('rfr', rfr)])

        param_grid = {
            "rfr__n_estimators": [300],
            "rfr__max_depth": [30],
            "rfr__min_samples_split": [2],
            "rfr__min_samples_leaf": [1],
            "rfr__max_features": ["sqrt"],
            "rfr__bootstrap": [False]
        }

        grid = GridSearchCV(estimator=pipe,
                            param_grid=param_grid,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        grid.fit(X_train, y_train)
        pred = grid.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        residual = y_test - pred

        score = (f'random forest regressor:\nr2: {r2}\nmae: {mae}\nmse: {mse}\nrmse: {rmse}\n')

        decision_df = pd.DataFrame({'actual': y_test,
                                    'predict': pred,
                                    'residual': residual}).reset_index(drop=True)


        plt.figure(figsize=(10, 8))
        plt.plot(decision_df['actual'], decision_df['actual'], c='green')
        plt.scatter(decision_df['actual'], decision_df['predict'], c='red')
        plt.title('RANDOM FOREST REGRESSOR\nACTUAL VS PREDICT')
        plt.ylabel('PREDICT')
        plt.xlabel('ACTUAL')
        plt.savefig('../plots/random_actual_predict.png',dpi=250,bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.title('RANDOM FOREST REGRESSOR\nPREDICT VS RESIDUAL')
        plt.plot(decision_df['predict'], decision_df['predict'], c='green')
        plt.scatter(decision_df['predict'], decision_df['residual'], c='red')
        plt.ylabel('RESIDUAL')
        plt.xlabel('PREDICT')
        plt.savefig('../plots/random_predict_residual.png', dpi=250, bbox_inches='tight')
        plt.show()

        estimator = grid.best_estimator_

        feature_importance = estimator.named_steps['rfr'].feature_importances_

        importance_df = estimator.named_steps['preprocessor'].get_feature_names_out()

        feature_df = pd.DataFrame({'feature': importance_df,
                                   'importance': feature_importance})

        feature_df = feature_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

        plt.figure(figsize=(10, 8))
        plt.title('RANDOM FOREST REGRESSOR\nFEATURE IMPORTANCE')
        plt.barh(feature_df['feature'].head(5), feature_df['importance'].head(5))
        plt.ylabel('FEATURE')
        plt.xlabel('RANGE')
        plt.gca().invert_yaxis()
        plt.savefig('../plots/random_feature_importance.png', dpi=250, bbox_inches='tight')
        plt.show()

        return score, decision_df, feature_df


if __name__ == '__main__':
    reg = Regressor('c:/extra_tree/loader/extra_tree_process.pkl')
    score, decision_df, feature_df = reg.decision()
    print(score)
    print(decision_df.head())
    print(feature_df.to_string())

    score, decision_df, feature_df = reg.extra()
    print(score)
    print(decision_df.head())
    print(feature_df.to_string())

    score, decision_df, feature_df = reg.random()
    print(score)
    print(decision_df.head())
    print(feature_df.to_string())





