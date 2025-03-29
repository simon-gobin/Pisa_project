from logging.handlers import RotatingFileHandler
import logging
import os
from cuml.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mean_squared_errorSK
from sklearn.metrics import mean_absolute_error as mean_absolute_errorSK
import matplotlib.pyplot as plt
import seaborn as sns
from cuml.ensemble import RandomForestRegressor  # RAPIDS GPU-based RF
from cuml.svm import SVR
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from cuml.metrics import r2_score as cuml_r2_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import explained_variance_score, median_absolute_error
from datetime import date
import json
from cuml.linear_model import ElasticNet


class bench_mark():


    def __init__(self, X, y, log_level=logging.INFO, top_n=15, model_penalty = 0.05):

        self.X = X
        self.y = y
        self.top_n = top_n
        self.model_penalty = model_penalty

        # Configure logging
        project_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(project_root, "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)


        log_filename = os.path.join(log_dir, f"benchmark_debug_{date.today()}.log")
        self.logger = logging.getLogger('ETLLogger')
        self.logger.setLevel(log_level)  # DEBUG or INFO for fewer messages

        if not self.logger.handlers:
            # Create a rotating file handler: 200 MB per file, up to 5 backups
            handler = RotatingFileHandler(log_filename, maxBytes=200 * 1024 * 1024, backupCount=5)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


        self.logger.info("ETL instance created.")
        self.results = {}

    def log(self, message):
        self.logger.info(message)
        print(message)  # Commented out to log only to file

    @staticmethod
    def plot(y_test, y_pred, model):
        # Compute errors
        errors = y_pred - y_test
        abs_errors = np.abs(errors)

        # Create a DataFrame for visualization
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': errors, 'Absolute Error': abs_errors})

        plt.figure(figsize=(10, 5))

        # Sort by actual scores for better visualization
        df_sorted = df.sort_values(by="Actual").reset_index(drop=True)

        plt.plot(df_sorted.index, df_sorted['Actual'], label="Actual", color='red', marker='o')
        plt.plot(df_sorted.index, df_sorted['Predicted'], label="Predicted", color='blue', linestyle='dashed',
                 marker='x')

        plt.xlabel(f"Student Index {model}")
        plt.ylabel("Score")
        plt.title(f"Actual vs. Predicted {model}")
        plt.legend()
        plt.show()

        # 🔹 Step 7: Box Plot - Absolute Errors by Student Score Ranges
        df['Score Bin'] = pd.cut(df['Actual'], bins=np.arange(0, 101, 10))  # Binning scores
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['Score Bin'], y=df['Error'])
        plt.xticks(rotation=45)
        plt.xlabel(f"{model}")
        plt.ylabel("Absolute Prediction Error")
        plt.title(f"Absolute Error Spread {model}")
        plt.show()


    def bayesian_optimization_ElasticNet(self):
        model_name= 'ElasticNet'
        def ElasticNet_evaluate(alpha, l1_ratio):
            params = {
                'alpha': alpha,
                'l1_ratio': l1_ratio
            }

            start_time_total = time.time()

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
            model = ElasticNet(**params, n_streams=1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = cuml_r2_score(y_test, y_pred)

            # Penalization for training time
            total_time = time.time() - start_time_total
            score = accuracy - (self.model_penalty * total_time)
            print(f'Accuracy = {accuracy:.4f} for training time {total_time:.2f}s')
            return score

        # Define parameter bounds
        param_bounds = {
            "alpha": (0.01, 5),
            "l1_ratio": (0.0, 1.0)
        }

        # Bayesian Optimization
        optimizer = BayesianOptimization(f=ElasticNet_evaluate, pbounds=param_bounds, random_state=42, verbose=2)
        optimizer.maximize(init_points=3, n_iter=5)

        # Best hyperparameters
        best_params = optimizer.max['params']
        print(f"Best hyperparameters: {best_params}")

        # Convert cuDF to pandas
        X_pandas = self.X.to_pandas()
        y_pandas = self.y.to_pandas()

        # KFold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mean_absolute_error_, r2_score_scores_, MAPE_, times_, MAE_ = [], [], [], [], []

        for train_index, test_index in kf.split(X_pandas):
            start_time_total = time.time()

            X_train, X_test = X_pandas.iloc[train_index], X_pandas.iloc[test_index]
            y_train, y_test = y_pandas.iloc[train_index], y_pandas.iloc[test_index]

            model = ElasticNet(**best_params, n_streams=1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            total_time = time.time() - start_time_total
            y_test = y_test.to_numpy()
            y_pred = y_pred.to_numpy()

            mean_absolute_error_met = mean_squared_errorSK(y_test, y_pred)
            r2_score_met = explained_variance_score(y_test, y_pred)
            MAPE_met = mean_absolute_errorSK(y_test, y_pred)
            MAE_met = median_absolute_error(y_test, y_pred)

            self.plot(y_test, y_pred, model_name)

            print(
                f'Mean squared error = {mean_absolute_error_met:.4f}, R² = {r2_score_met:.4f}, MAPE = {MAPE_met:.4f}, '
                f'Median AE = {MAE_met:.4f}, Training Time = {total_time:.2f}s')

            mean_absolute_error_.append(mean_absolute_error_met)
            r2_score_scores_.append(r2_score_met)
            MAPE_.append(MAPE_met)
            times_.append(total_time)
            MAE_.append(MAE_met)

        # Aggregate scores
        avg_mse = np.mean(mean_absolute_error_)
        avg_r2 = np.mean(r2_score_scores_)
        avg_mape = np.mean(MAPE_)
        avg_mae = np.mean(MAE_)
        avg_time = np.mean(times_)

        print(
            f"Mean Scores for ElasticNet CV: MSE = {avg_mse:.4f}, R² = {avg_r2:.4f}, MAPE = {avg_mape:.4f}, MAE = {avg_mae:.4f}, Time = {avg_time:.2f}s")

        self.results["ElasticNet"] = {
            "mean_squared_error": avg_mse,
            "r2_score": avg_r2,
            "MAPE": avg_mape,
            "MAE": avg_mae,
            "training_time": avg_time,
            "best_params": best_params
        }

    def bayesian_optimization_rf(self):
        model_name = 'Random forest'
        def rf_evaluate(n_estimators, max_depth, max_features):
            params = {
                'n_estimators': int(n_estimators),
                'max_depth': int(max_depth),
                'max_features': max_features
            }

            start_time_total = time.time()

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
            model = RandomForestRegressor(**params, n_streams=1, random_state=42)
            model.fit(X_train, y_train)

            # Predict and calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = cuml_r2_score(y_test, y_pred)

            # Penalization for training time
            total_time = time.time() - start_time_total
            score = accuracy - (self.model_penalty * total_time)
            print(f'Accuracy = {accuracy:.4f} for training time {total_time: .2f}')
            return score

        # Define parameter bounds for Bayesian Optimization
        param_bounds = {
            "n_estimators": (50, 200),
            "max_depth": (3, 15),
            "max_features": (0.5, 1.0)
        }

        # Perform Bayesian Optimization
        optimizer = BayesianOptimization(f=rf_evaluate, pbounds=param_bounds, random_state=42, verbose=2)
        optimizer.maximize(init_points=3, n_iter=5)

        # Get the best hyperparameters
        best_params = optimizer.max['params']
        print(f"Best hyperparameters: {best_params}")

        # Convert parameter values to integers where required
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])

        # Convert cuDF to pandas for compatibility with sklearn
        X_pandas = self.X.to_pandas()
        y_pandas = self.y.to_pandas()

        # Initialize KFold and accuracy list
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mean_absolute_error_, r2_score_scores_, MAPE_, times_, MAE_ = [], [], [], [], []

        for train_index, test_index in kf.split(X_pandas):
            start_time_total = time.time()
            # Split the data into train and test
            X_train, X_test = X_pandas.iloc[train_index], X_pandas.iloc[test_index]
            y_train, y_test = y_pandas.iloc[train_index], y_pandas.iloc[test_index]

            # Retrain model with best hyperparameters
            model = RandomForestRegressor(**best_params, n_streams=1, random_state=42)
            model.fit(X_train, y_train)

            # Predict and calculate accuracy
            y_pred = model.predict(X_test)

            total_time = time.time() - start_time_total
            y_test = y_test.to_numpy()
            y_pred = y_pred.to_numpy()
            mean_absolute_error_met = mean_squared_errorSK(y_test, y_pred)
            r2_score_met = explained_variance_score(y_test, y_pred)
            MAPE_met = mean_absolute_errorSK(y_test, y_pred)
            MAE_met = median_absolute_error(y_test, y_pred)
            self.plot(y_test, y_pred, model_name)

            print(
                f'Mean square error = {mean_absolute_error_met:.4f}, r2_score_met = {r2_score_met}, MAPE_met = {MAPE_met}, Mean absoluterror = {MAE_met}, for training time {total_time: .2f}')

            mean_absolute_error_.append(mean_absolute_error_met)
            r2_score_scores_.append(r2_score_met)
            MAPE_.append(MAPE_met)
            times_.append(total_time)
            MAE_.append(MAE_met)

        # Calculate mean accuracy across folds
        avg_mean_absolute_error = sum(mean_absolute_error_) / len(mean_absolute_error_)
        avg_r2_score_scores = sum(r2_score_scores_) / len(r2_score_scores_)
        mean_MAPE = sum(MAPE_) / len(MAPE_)
        mean_MAE = sum(MAE_) / len(MAE_)
        mean_time = sum(times_) / len(times_)
        print(
            f"Mean Scores for Cross-Validation: mean_square_error =  {avg_mean_absolute_error:.4f}, r2_score_scores =  {avg_r2_score_scores:.4f},"
            f" MAPE = {mean_MAPE:.4f}, MAE = {mean_MAE}, time = {mean_time:.2f}")

        self.results["RF"] = {
            "mean_squared_error": avg_mean_absolute_error,
            "r2_score": avg_r2_score_scores,
            "MAPE": mean_MAPE,
            "MAE": mean_MAE,
            "training_time": mean_time,
            "best_params": best_params
        }

    def bayesian_optimization_SVR(self):
        model_name = 'SVR'
        def svr_evaluate(C, epsilon):
            params = {
                'C': float(C),
                'epsilon': float(epsilon)
            }

            start_time_total = time.time()

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            y_train = y_train.astype('float32')
            y_test = y_test.astype('float32')
            model = SVR(**params, kernel='rbf', gamma='scale')
            model.fit(X_train, y_train)

            # Predict and calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = cuml_r2_score(y_test, y_pred)
            # Penalization for training time
            total_time = time.time() - start_time_total
            score = accuracy - (self.model_penalty * total_time)
            print(f'Accuracy = {accuracy:.4f} for training time {total_time: .2f}')
            return score

        # Define parameter bounds for Bayesian Optimization
        param_bounds = {
            "C": (0.1, 10),
            "epsilon": (0.01, 1)
        }

        # Perform Bayesian Optimization
        optimizer = BayesianOptimization(f=svr_evaluate, pbounds=param_bounds, random_state=42, verbose=2)
        optimizer.maximize(init_points=3, n_iter=5)

        # Get the best hyperparameters
        best_params = optimizer.max['params']
        print(f"Best hyperparameters: {best_params}")

        # Convert cuDF to pandas for compatibility with sklearn
        X_pandas = self.X.to_pandas()
        y_pandas = self.y.to_pandas()

        # Initialize KFold and accuracy list
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mean_absolute_error_, r2_score_scores_, MAPE_, times_, MAE_ = [], [], [], [], []

        for train_index, test_index in kf.split(X_pandas):
            start_time_total = time.time()
            # Split the data into train and test
            X_train, X_test = X_pandas.iloc[train_index], X_pandas.iloc[test_index]
            y_train, y_test = y_pandas.iloc[train_index], y_pandas.iloc[test_index]
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            y_train = y_train.astype('float32')
            y_test = y_test.astype('float32')

            # Retrain model with best hyperparameters
            model = SVR(**best_params, kernel='rbf', gamma='scale')
            model.fit(X_train, y_train)

            # Predict and calculate accuracy
            y_pred = model.predict(X_test)

            total_time = time.time() - start_time_total
            y_test = y_test.to_numpy()
            y_pred = y_pred.to_numpy()
            mean_absolute_error_met = mean_squared_errorSK(y_test, y_pred)
            r2_score_met = explained_variance_score(y_test, y_pred)
            MAPE_met = mean_absolute_errorSK(y_test, y_pred)
            MAE_met = median_absolute_error(y_test, y_pred)
            self.plot(y_test, y_pred, model_name)

            print(
                f'Mean square error = {mean_absolute_error_met:.4f}, r2_score_met = {r2_score_met}, MAPE_met = {MAPE_met}, Mean absoluterror = {MAE_met}, for training time {total_time: .2f}')

            mean_absolute_error_.append(mean_absolute_error_met)
            r2_score_scores_.append(r2_score_met)
            MAPE_.append(MAPE_met)
            times_.append(total_time)
            MAE_.append(MAE_met)

        # Calculate mean accuracy across folds
        avg_mean_absolute_error = sum(mean_absolute_error_) / len(mean_absolute_error_)
        avg_r2_score_scores = sum(r2_score_scores_) / len(r2_score_scores_)
        mean_MAPE = sum(MAPE_) / len(MAPE_)
        mean_MAE = sum(MAE_) / len(MAE_)
        mean_time = sum(times_) / len(times_)
        print(
            f"Mean Scores for Cross-Validation: mean_square_error =  {avg_mean_absolute_error:.4f}, r2_score_scores =  {avg_r2_score_scores:.4f}, "
            f"MAPE = {mean_MAPE:.4f},MAE = {mean_MAE}, time = {mean_time:.2f}")

        self.results["SVR"] = {
            "mean_squared_error": avg_mean_absolute_error,
            "r2_score": avg_r2_score_scores,
            "MAPE": mean_MAPE,
            "MAE": mean_MAE,
            "training_time": mean_time,
            "best_params": best_params
        }

    def bayesian_optimization_XGB(self):
        model_name = 'XGBoost'
        def XGB_evaluate(n_estimators, max_depth, max_features):
            # Convert float hyperparameters to integers
            max_depth = int(round(max_depth))
            n_estimators = int(round(n_estimators))

            # Define additional hyperparameters
            learning_rate = 0.1
            subsample = 0.8
            colsample_bytree = max_features  # Use optimized max_features

            # XGBoost parameters
            params = {
                'tree_method': "hist",
                'device': "cuda",
                'random_state' : 42,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree
            }
            start_time_total = time.time()

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)

            # Drop duplicated columns first
            dups = X_train.columns[X_train.columns.duplicated()].tolist()
            if dups:
                print("❗ Duplicated columns dropped:", dups)
            X_train = X_train.loc[:, ~X_train.columns.duplicated()]
            X_test = X_test.loc[:, ~X_test.columns.duplicated()]

            # THEN sanitize the names
            X_train.columns = X_train.columns.astype(str).str.replace(r"[\[\]<>]", "", regex=True).str.replace(" ", "_")
            X_test.columns = X_test.columns.astype(str).str.replace(r"[\[\]<>]", "", regex=True).str.replace(" ", "_")

            # Train XGBoost model
            model = xgb.XGBRegressor(**params, n_estimators=n_estimators)
            model.fit(X_train, y_train)

            # Predict and calculate accuracy
            preds = model.predict(X_test)
            accuracy = cuml_r2_score(y_test, preds)


            # Penalization for training time
            total_time = time.time() - start_time_total
            score = accuracy - (self.model_penalty * total_time)
            print(f'R² = {accuracy:.4f} for training time {total_time:.2f}')

            return score

        # Define parameter bounds for Bayesian Optimization
        param_bounds = {
            "n_estimators": (50, 200),
            "max_depth": (3, 15),
            "max_features": (0.5, 1.0)
        }

        # Perform Bayesian Optimization
        optimizer = BayesianOptimization(f=XGB_evaluate, pbounds=param_bounds, random_state=42, verbose=2)
        optimizer.maximize(init_points=3, n_iter=5)

        # Get the best hyperparameters
        best_params = optimizer.max['params']
        print(f"Best hyperparameters: {best_params}")

        # Convert parameter values to integers where required
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])

        # Convert cuDF to pandas
        X_pandas = self.X.to_pandas()
        y_pandas = self.y.to_pandas()

        # Initialize KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mean_absolute_error_, r2_score_scores_, MAPE_, times_, MAE_ = [], [], [], [], []

        for train_index, test_index in kf.split(X_pandas):
            start_time_total = time.time()

            # Split the data into train and test
            X_train, X_test = X_pandas.iloc[train_index], X_pandas.iloc[test_index]
            y_train, y_test = y_pandas.iloc[train_index], y_pandas.iloc[test_index]

            # Drop duplicated columns first
            dups = X_train.columns[X_train.columns.duplicated()].tolist()
            if dups:
                print("❗ Duplicated columns dropped:", dups)
            X_train = X_train.loc[:, ~X_train.columns.duplicated()]
            X_test = X_test.loc[:, ~X_test.columns.duplicated()]

            # THEN sanitize the names
            X_train.columns = X_train.columns.astype(str).str.replace(r"[\[\]<>]", "", regex=True).str.replace(" ", "_")
            X_test.columns = X_test.columns.astype(str).str.replace(r"[\[\]<>]", "", regex=True).str.replace(" ", "_")

            # Retrain model with best hyperparameters
            model = xgb.XGBRegressor(**best_params, tree_method="hist", device="cuda", random_state=42)
            model.fit(X_train, y_train)

            # Predict and calculate accuracy
            y_pred = model.predict(X_test)

            total_time = time.time() - start_time_total

            mean_absolute_error_met = mean_squared_errorSK(y_test, y_pred)
            r2_score_met = explained_variance_score(y_test, y_pred)
            MAPE_met = mean_absolute_errorSK(y_test, y_pred)
            MAE_met = median_absolute_error(y_test, y_pred)
            # plot
            # Make sure the model is fitted before calling get_booster
            booster = model.get_booster()

            # Plot top N features by importance (gain)
            plt.figure(figsize=(10, 6))
            ax = xgb.plot_importance(
                booster,
                importance_type='gain',  # Or 'weight' or 'cover'
                max_num_features= self.top_n,
                height=0.4,
                show_values=True,
                values_format="{v:.2f}",
                xlabel='Gain',
                title=f'Top {self.top_n} Important Features (Gain)',
                grid=True
            )
            plt.tight_layout()
            plt.show()
            self.plot(y_test, y_pred, model_name)

            print(
                f'Mean squared error = {mean_absolute_error_met:.4f}, R² = {r2_score_met:.4f}, RMSE = {MAPE_met:.4f}, Median AE = {MAE_met:.4f}, Training Time = {total_time:.2f}s')

            mean_absolute_error_.append(mean_absolute_error_met)
            r2_score_scores_.append(r2_score_met)
            MAPE_.append(MAPE_met)
            times_.append(total_time)
            MAE_.append(MAE_met)

        # Calculate mean accuracy across folds
        avg_mae = np.mean(mean_absolute_error_)
        avg_r2 = np.mean(r2_score_scores_)
        avg_rmse = np.mean(MAPE_)
        avg_mae_median = np.mean(MAE_)
        avg_time = np.mean(times_)

        print(
            f"Mean Scores for Cross-Validation: mean_square_error =  {avg_mae:.4f}, r2_score_scores =  {avg_r2:.4f}, "
            f"MAPE = {avg_rmse:.4f}, MAE = {avg_mae_median:.4f}, time = {avg_time:.2f}")

        self.results["XGB"] = {
            "mean_squared_error": avg_mae,
            "r2_score": avg_r2,
            "MAPE": avg_rmse,
            "MAE": avg_mae_median,
            "training_time": avg_time,
            "best_params": best_params
        }

    def run_all(self):
        print('Run XGBoost')
        self.bayesian_optimization_XGB()
        print('Run Random forest')
        self.bayesian_optimization_rf()
        print('Run SVR')
        self.bayesian_optimization_SVR()
        print('Run ElasticNet')
        self.bayesian_optimization_ElasticNet()

    def summarize_results(self):
        print("Summary of Model Performance:")
        for model, metrics in self.results.items():
            print(f"{model}")
            for metric, value in metrics.items():
                print(f" {metric}: {value}")

    def save_results(self, filename="benchmark_results.json"):
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=4)