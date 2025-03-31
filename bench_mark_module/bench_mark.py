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

from sklearn.feature_selection import (f_regression, r_regression, mutual_info_regression)

from cuml.decomposition import PCA
from cuml.preprocessing import StandardScaler


class bench_mark():


    def __init__(self, X, y, top_feature_percent = 1, log_level=logging.INFO, top_n=15, model_penalty = 0.05, top_feature_reduction = False, PCA = False):

        self.X = X
        self.y = y
        self.top_n = top_n
        self.model_penalty = model_penalty
        self.top_features = top_feature_percent
        self.top_feature_reduction = top_feature_reduction
        self.PCA = PCA

        if self.PCA == True and self.top_feature_reduction == True:
            self.logger.error('ERROR : PCA and feature reduction are both enabled')
            print('PCA and feature reduction are both enabled')

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
        #print(message)  # Commented out to log only to file

    def EDA(self):
        X_panda = self.X.to_pandas()
        y_panda = self.y.to_pandas()

        f_values, P_value = f_regression(X_panda, y_panda)
        mutual_info = mutual_info_regression(X_panda, y_panda)
        Correlation = r_regression(X_panda, y_panda)

        # Create a DataFrame from the results
        results_df = pd.DataFrame({
            'Feature': self.X.columns,
            'P_value': P_value,
            'f_values': f_values,
            'Correlation': Correlation,
            'Correlation abs' : abs(Correlation),
            'Mutual_info': mutual_info
        })

        # Sort and get top Correlation
        num_features = len(X_panda.columns)
        top_percent_cutoff = int(self.top_features * num_features)

        #filter the result for the plot
        top_features_mutual_info = results_df.nlargest(top_percent_cutoff, 'Mutual_info')['Feature'].tolist()
        top_features_corr = results_df.nlargest(top_percent_cutoff, 'Correlation abs')['Feature'].tolist()
        top_features_fvalues = results_df.nlargest(top_percent_cutoff, 'f_values')['Feature'].tolist()
        # Combine all top features into one list and remove duplicates
        top_features = list(set(top_features_mutual_info  + top_features_corr + top_features_fvalues))

        df_top = results_df[results_df['Feature'].isin(top_features)].copy()

        # Print the combined list of top features
        logging.info(f'Combined list of unique top features: {top_features}')

        with open("top_features.json", "w") as f:
            json.dump(top_features, f, indent=4)



        #plot
        plt.figure(figsize=(62, 10))

        plt.subplot(3, 1, 1)
        sns.barplot(y='f_values', x=df_top, data=df_top)
        plt.title('P-values of Features')

        plt.subplot(3, 1, 2)
        sns.barplot(y='Correlation', x=df_top, data=df_top)
        plt.title('Correlation of Features with Target')


        plt.subplot(3, 1, 3)
        sns.barplot(y='Mutual info', x=df_top, data=df_top)
        plt.title('Importance Decision Tree')

        plt.tight_layout()
        plt.show()
        plt.savefig('mutliplot_3.jpg')

        # Plot the heatmap
        plt.figure(figsize=(15, 12))
        sns.heatmap(df_top, annot=True, cmap='coolwarm', center=0, fmt=".2f")
        plt.title("Feature Correlation Heatmap (Top Correlated Features)")
        plt.show()

        if self.top_feature_reduction is True:
            self.X = self.X[top_features]

        if self.PCA is True:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X)
            pca = PCA()
            pca.fit(X_scaled)
            explained_variance = pca.explained_variance_ratio_

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(cp.asnumpy(cp.cumsum(explained_variance)), marker='o')
            plt.grid(axis="both")
            plt.xlabel("Principal Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title("Cumulative Explained Variance by Principal Components")
            sns.despine()
            plt.tight_layout()
            plt.show()

            pca = PCA(n_components=0.95)
            self.X = pca.fit_transform(X_scaled)
            self.logger.info(f" PCA reduced to {X_pca.shape[1]} components...")

        return self.X



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
        # Create name mapping before renaming
        column_name_map = {
            col: f"col_{i}" for i, col in enumerate(self.X.columns)
        }


        # Rename columns using the map
        self.X_renamed = self.X.rename(columns=column_name_map)

        # Save the reversed map for later (used for plotting)
        reverse_column_map = {v: k for k, v in column_name_map.items()}
        self.column_map = reverse_column_map

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
            X_train, X_test, y_train, y_test = train_test_split(self.X_renamed , self.y, test_size=0.33, random_state=42)


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
        X_pandas = self.X_renamed .to_pandas()
        y_pandas = self.y.to_pandas()

        # Initialize KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mean_absolute_error_, r2_score_scores_, MAPE_, times_, MAE_ = [], [], [], [], []

        for train_index, test_index in kf.split(X_pandas):
            start_time_total = time.time()

            # Split the data into train and test
            X_train, X_test = X_pandas.iloc[train_index], X_pandas.iloc[test_index]
            y_train, y_test = y_pandas.iloc[train_index], y_pandas.iloc[test_index]

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
            # Rename features in booster
            booster = model.get_booster()
            feature_scores = booster.get_score(importance_type='gain')

            # Map back to original names
            feature_scores_readable = {
                reverse_column_map.get(feat, feat): score
                for feat, score in feature_scores.items()
            }

            # Convert to a sorted list of tuples
            sorted_features = sorted(feature_scores_readable.items(), key=lambda x: x[1], reverse=True)

            # Plot manually using matplotlib
            top_n = self.top_n
            top_features = sorted_features[:top_n]
            labels, gains = zip(*top_features)

            plt.figure(figsize=(10, 6))
            plt.barh(range(top_n), gains[::-1])
            plt.yticks(range(top_n), labels[::-1])
            plt.xlabel("Gain")
            plt.title(f"Top {top_n} Important Features (Gain)")
            plt.grid(True)
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
        print('Run EDA')
        self.EDA()
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