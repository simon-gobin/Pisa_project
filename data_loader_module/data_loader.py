import os
import gc
import sqlalchemy
import numpy as np
import pandas as pd
import cudf
from cuml.preprocessing import LabelEncoder as gpu_encoder
from google.cloud.sql.connector import Connector
from google.colab import auth
from getpass import getpass


class loader:
    def __init__(self, X_query, y_query, index='Intl_School_ID', db_password=None, chunksize= 20000): # Intl_School_ID : school Intl_Student_ID:student
        self.X_query = X_query
        self.y_query = y_query
        self.index = index
        self.db_password = db_password or getpass()
        self.chunksize = chunksize
        auth.authenticate_user()

    def load(self):
        X_full = None

        # Initialize parameters
        INSTANCE_CONNECTION_NAME = 'pisasql:europe-west2:pisasql'
        print(f"Your instance connection name is: {INSTANCE_CONNECTION_NAME}")
        DB_USER = "postgres"
        DB_PASS = self.db_password
        DB_NAME = "pisa"

        # Initialize Connector object
        connector = Connector()

        # Function to return the database connection object
        def getconn():
            return connector.connect(
                INSTANCE_CONNECTION_NAME,
                "pg8000",
                user=DB_USER,
                password=DB_PASS,
                db=DB_NAME
            )

        # Create connection pool
        pool = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
        )

        # Load `y`
        y = pd.read_sql(self.y_query, pool)
        y = cudf.from_pandas(y)

        # Initialize GPU Label Encoder
        le = gpu_encoder()

        # Process `X` in chunks
        for chunk in pd.read_sql(self.X_query, pool, chunksize=self.chunksize):

            # Convert boolean-like strings to actual boolean
            for col in chunk.columns:
                chunk[col] = chunk[col].replace({'True': True, 'False': False, 'None': np.nan})

                try:
                    chunk[col] = chunk[col].astype('boolean')
                except Exception as e:
                    print(f"Skipping boolean conversion for {col}: {e}")

            # Convert to cuDF (GPU DataFrame)
            chunk = cudf.from_pandas(chunk)

            # Apply GPU Label Encoding
            for col in chunk.columns:
                if chunk[col].dtype == 'object':
                    chunk[col] = le.fit_transform(chunk[col])

            for col in chunk.columns:
                col_dtype = chunk[col].dtype
                if col_dtype == 'bool':
                    chunk[col] = chunk[col].fillna(False)
                else:
                    chunk[col] = chunk[col].fillna(0)

            # Concatenate processed chunks
            if X_full is None:
                X_full = chunk
            else:
                X_full = cudf.concat([X_full, chunk], ignore_index=True)

            # Explicitly delete chunk to free memory (optional)
            del chunk
            gc.collect()

        # Merge data frame to be sure have same lengh
        df = cudf.merge(X_full, y, left_on=self.index, right_on=self.index, how='inner')
        y = df['avg_composite_percentile']
        X = df.drop(columns=['avg_composite_percentile'])

        connector.close()

        # Debugging: Check object-type columns
        print("Object-type columns:", X_full.select_dtypes(include='object').columns)
        print("Total object-type columns:", len(X_full.select_dtypes(include='object').columns))
        print('finish')

        return X, y