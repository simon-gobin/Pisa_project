import pandas as pd
from sqlalchemy import create_engine
from google.cloud.sql.connector import Connector
from google.colab import auth
from google.colab import userdata
import logging
from google.cloud.sql.connector import Connector
import sqlalchemy
import pandas as pd





class bench_mark:
    def __init__(self,query, log_level=logging.INFO):

        self.query = query

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

    def data_loader(self.query):

        # initialize parameters
        INSTANCE_CONNECTION_NAME = 'pisasql:europe-west2:pisasql'  # i.e demo-project:us-central1:demo-instance
        print(f"Your instance connection name is: {INSTANCE_CONNECTION_NAME}")
        DB_USER = "postgres"
        DB_PASS = db_password
        DB_NAME = "pisa"

        # initialize Connector object
        connector = Connector()

        # function to return the database connection object
        def getconn():
            conn = connector.connect(
                INSTANCE_CONNECTION_NAME,
                "pg8000",
                user=DB_USER,
                password=DB_PASS,
                db=DB_NAME
            )
            return conn

        # create connection pool with 'creator' argument to our connection object function
        pool = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
        )

        df = pd.read_sql(self.query, pool)

        for col in df.columns:
            try:
                df[col] = df[col].replace({'True': True, 'False': False, 'None': np.nan})
                df[col] = df[col].astype('boolean')
                # self.logger.debug(f'{df[col].name}transform as boolean')
            except Exception as e:
        # self.logger.error(e)
        # self.logger.info(df_school_questions.head())

        return df







