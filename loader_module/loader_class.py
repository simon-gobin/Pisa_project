import os
from logging.handlers import RotatingFileHandler
from datetime import date
import logging
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database


class loader:

    def __init__(self, file_path, db_name, table_name, log_level=logging.INFO):
        self.file_path = file_path
        self.db_name = db_name
        self.table_name = table_name

        # Configure logging
        project_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(project_root, "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        log_filename = os.path.join(log_dir, f"{base_name}_loader_debug_{date.today()}.log")
        self.logger = logging.getLogger('ETLLogger')
        self.logger.setLevel(log_level)  # DEBUG or INFO for fewer messages

        if not self.logger.handlers:
            # Create a rotating file handler: 200 MB per file, up to 5 backups
            handler = RotatingFileHandler(log_filename, maxBytes=200 * 1024 * 1024, backupCount=5)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("ETL instance created.")

    def log(self, message):
        self.logger.info(message)
        print(message)  # Commented out to log only to file

    def load_data(self):

        try:
            df = pd.read_parquet(self.file_path)
            df = df.loc[:, ~df.columns.duplicated()]
            for col in df.columns:
                self.logger.debug(f"{col} is dtype {df[col].dtypes}")
            db_name = self.db_name
        except Exception as e:
            self.logger.error(f"Error loading file {self.file_path}: {e}")
            return

        base_url = 'postgresql://{0}:{1}@{2}:{3}'.format('postgres', 'password', 'localhost', '5432')
        url = f"{base_url}/{db_name}"
        engine = create_engine(url)
        table_name = self.table_name

        try:
            if not database_exists(engine.url):
                create_database(engine.url)
                self.logger.info(f"Database '{db_name}' created.")
            else:
                self.logger.info(f"Database '{db_name}' already exists.")

            self.logger.info(f"Database exists: {database_exists(engine.url)}")
        except Exception as e:
            self.logger.error(f"Error checking/creating database: {e}")
            return

        try:
            # Use SQLAlchemy to write the DataFrame
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            self.logger.info(f'Data successfully inserted into table {table_name}')
            from sqlalchemy import inspect

            inspector = inspect(engine)
            columns = inspector.get_columns(self.table_name)
            for column in columns:
                self.logger.debug(f"PostgreSQL Column: {column['name']}, Data type: {column['type']}")

        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
        finally:
            engine.dispose()