import pyreadstat
import numpy as np
import sqlalchemy
import openai
import pandas as pd
import os
import gc
import pyarrow as pa
import pyarrow.parquet as pq
import re
import logging
from logging.handlers import RotatingFileHandler
from datetime import date


class ETL:

    def __init__(self, input_file, output_dir='cleaned_out', batch_size=50000,
                 row_limit=None, verbose=True, drop_empty=True, short_name=False, student_score=False, log=True,
                 log_level=logging.INFO):

        self.input_file = input_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.row_limit = row_limit
        self.verbose = verbose
        self.logger = log
        self.drop_empty = drop_empty
        self.short_name = short_name
        self.student_score = student_score
        self.final_schema_columns = None
        self.final_dtypes = None
        self.first_batch_columns = []
        self.relevant_column = []
        self.int_col = []  # List of column names converted to int
        self.bool_col = []  # List of column names converted to boolean
        self.empty_col_names = []  # List of column names that are entirely empty
        self.shortened_mapping = {}  # To store the shortened mapping for column names
        self.batch_number = 1
        self.processed_rows = 0
        self.nul_vall = {95.0: np.nan, 96.0: np.nan, 97.0: np.nan, 98.0: np.nan,
                         99.0: np.nan, 9.0: np.nan, 8.0: np.nan, 7.0: np.nan,
                         6.0: np.nan}  # e.g. 6.0: 'Not Reached', etc.

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(f'file created: {output_dir}')

        # Configure logging
        import datetime
        import logging
        from logging.handlers import RotatingFileHandler

        project_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(project_root, "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        log_filename = os.path.join(log_dir, f"{base_name}_etl_debug_{date.today()}.log")

        self.logger = logging.getLogger('ETLLogger')
        self.logger.setLevel(log_level)  # DEBUG or INFO for fewer messages

        if not self.logger.handlers:
            # Create a rotating file handler: 200 MB per file, up to 5 backups
            handler = RotatingFileHandler(log_filename, maxBytes=200 * 1024 * 1024, backupCount=5)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if self.verbose:
            self.logger.info("ETL instance created.")
            # self.logger.propagate = False

    def log(self, message):
        if self.verbose:
            self.logger.info(message)
            # print(message)  # Commented out to log only to file

    def generate_shortened_columns(self, df, max_length=64, api_key='OPENAI_API_KEY', verbose=True):
        from openai import OpenAI
        mapping = {}
        client = OpenAI(api_key=os.getenv(api_key))
        column_names = df.columns

        for col in column_names:
            if len(col) > max_length:
                prompt = (
                    f"Shorten the following column name to be concise, meaningful, "
                    f"and less than {max_length} characters. Include unique identifiers "
                    f"like '1', '2', etc., to differentiate it from similar columns:\n\n"
                    f"Original column name: {col}"
                )
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",  # Use a valid model name
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    shortened_name = completion.choices[0].message.content.strip()
                    if verbose:
                        self.logger.info(f"Suggested shortened name for '{col}': {shortened_name}")
                    shortened_name = shortened_name[:max_length]
                    counter = 1
                    while shortened_name in mapping.values():
                        shortened_name = f"{shortened_name[:max_length - len(str(counter)) - 1]}_{counter}"
                        counter += 1
                    mapping[col] = shortened_name
                    if self.verbose:
                        self.logger.info(f"Final shortened name for '{col}': {shortened_name}")
                except Exception as e:
                    self.logger.error(f"Error processing column '{col}': {str(e)}")
            else:
                mapping[col] = col
                if self.verbose:
                    self.logger.info(f"'{col}' is already within the max length.")

        # Final duplicate check (for debugging purposes)
        duplicates = {}
        for key, value in mapping.items():
            duplicates.setdefault(value, []).append(key)
        final_duplicates = {key: values for key, values in duplicates.items() if len(values) > 1}
        if final_duplicates and self.verbose:
            self.logger.warning("Duplicate column names remain after resolution:")
            for dup_name, orig_cols in final_duplicates.items():
                self.logger.warning(f"Duplicate shortened name: {dup_name} -> Original columns: {orig_cols}")

        return mapping

    def string_tran(self, df):
        # Define your mapping: keys are tuples of substrings, values are the target output.
        mapping = {
            ('none', 'nan'): np.nan,

            ('not', 'no', 'never', 'neither trained', 'other country', '0', 'no credit', 'incorrect',
             'less than isced Level 1', "i don't know what this is", 'this did not happen to me',
             'not at all confident', "i don't know.", "don't", '0 days', '0 hours', 'approximately equal',
             'not checked', 'general teacher', 'poor', 'not at all satisfied'): 0,

            ('once', 'likely', 'sometime', 'strongly disagree', 'yes', 'checked', 'content overlap',
             'trained to teach', 'once or twice a year', 'studied aboard', 'seldom', 'very little importance',
             'less than half of the lessons', 'rarely', 'i attended', 'low', 'country of test', 'little emphasis',
             'little', 'overview or introduction to topic', '1 to 10', 'one other school', 'one', 'small',
             'few times a year', '15 students or fewer', '1 to 10', 'full credit', 'partial credit', 'correct',
             'permanent employment', '1 time', 'isced level 1', 'i cannot do this', 'not at all upset',
             'less than 1 hour a day', 'about once or twice a month', 'not very confident', 'a few times',
             'less than half of the time', 'heard of it once or twice', 'up to 30 minutes a day', '1 time',
             'somewhat important', "heard of it, but I don't recall the meaning", 'well prepared', '1 day',
             'one or two times', '1-5', '1 or 2', 'one', 'language of the test', '1-10 books', 'i learnt less when',
             'up to 10 hours', 'native student', 'village', 'less than half', '1', 'mostly low',
             'part-time (less than 50 of full-time hours)', 'mathematics teacher', 'fair', 'not satisfied',
             'very difficult', 'more than seven days ago'): 1,

            ('disagree', 'quite a bit', 'once or twice a month', 'often', 'some', 'about half of the lessons',
             'sometimes', 'moderate', '2-4 times a year', 'it was an area of emphasis', '11 to 30',
             'two or more other', 'two', 'moderate', 'few times a month', '16-20 students', '11 to 20', 'medium Level',
             '2 times', 'isced level 2', 'i struggle to do this on my own', 'a little upset',
             'between 1 and 3 hours a day', 'about once or twice a week', 'confident', 'about once or twice a week',
             'about half of the time', 'heard of it a few times', 'more than 30 minutes and up to 1 hour a day',
             '2 times', 'important', 'learnt about it, and I know what it means', 'very well prepared', '2 days',
             'three or four times', '6-10', '3 - 5', 'two', 'other language', '11-25 books', 'i learnt about as much',
             '11-20 hours', 'first-Generation student', 'small town', 'about half', '2', 'mostly average',
             'part-time (50-70 of full-time hours)', 'good', 'satisfied', 'difficult'): 2,

            ('agree', 'lot', 'once or twice a week', 'very often', 'always', 'many', 'more than half of the lessons',
             'frequently', 'high', '5-10 times a year', 'most', '31 to 60', 'large', 'once a week or more',
             '21-25 students', '21 to 30', '3 times', 'isced level 3.3', 'i can do with a bit of effort', 'quite upset',
             'more than 3 hours and up to 5 hours a day', 'every day or almost every day', 'very confident',
             'more than half of the time', 'heard of it often', 'more than 1 hour and up to 2 hours a day', '3 times',
             'very important', '3 days', 'five or more times', 'more than 10', 'more than 5', 'three', '26-100 books',
             'i learnt more', '21-30 hours', 'second-Generation student', 'town', 'more than half', '3', 'mostly high',
             'part-time (71-90 of full-time hours)', 'excellent', 'totally satisfied', 'easy', 'today or yesterday',
             'extremely'): 3,

            ('strongly agree', 'every day or almost every day', 'full control', 'every', '1-3 times a month',
             'all', 'more than 60', '26-30 students', '31 to 40', '4 times', 'isced level 3.4', 'i can easily do this',
             'very upset', 'more than 5 hours and up to 7 hours a day', 'several times a day',
             'all or almost all of the time', 'know it well, understand the concept',
             'more than 2 hours and up to 3 hours a day', '4 times', '4 days', 'four', '101-200 books', '31-40 hours',
             'city', 'all or almost all', '4', '31-35 students', 'full-time (more than 90 of full-time hours)',
             'very easy'): 4,

            ('81 to 90', '5 times', 'isced level 8', 'more than 3 hours and up to 4 hours a day', 'isced level 4',
             'five', 'large city', '5', '36-40 students'): 5,

            ('91 to 100', '10 times', '10 or more', 'six', '6', '41-45 students'): 6,

            ('41-45 students', '7', '46-50 students'): 7,

            ('more than 50 students', '8'): 8
        }

        def transform_value(val):
            if pd.isnull(val):
                return np.nan
            val = str(val).replace('%', '').strip().lower()

            # Build a flattened list of (keyword, new_val) pairs.
            keyword_mapping = []
            for keywords, new_val in mapping.items():
                for keyword in keywords:
                    keyword_clean = keyword.replace('%', '').strip().lower()
                    if keyword_clean:
                        keyword_mapping.append((keyword_clean, new_val))

            # Sort the list by length of keyword descending
            keyword_mapping.sort(key=lambda x: len(x[0]), reverse=True)

            # exact match first to reduce computing time
            for keyword_clean, new_val in keyword_mapping:
                if val == keyword_clean:
                    self.logger.debug(f"Exact match: Value '{val}' matched keyword '{keyword_clean}' -> {new_val}")
                    return new_val

            # Check each keyword
            for keyword_clean, new_val in keyword_mapping:
                pattern = r'\b' + re.escape(keyword_clean) + r'\b'
                if re.search(pattern, val):
                    self.logger.debug(f"Value '{val}' matched keyword '{keyword_clean}' -> {new_val}")
                    return new_val
            return val

        int_cols = []
        # Select only object columns
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            original_values = df[col].copy()
            try:
                original_unique = df[col].unique()
                df[col] = df[col].apply(transform_value)
                if self.verbose and self.batch_number == 1:
                    self.logger.info(
                        f"Column '{col}': original unique values {original_unique} replaced by {df[col].unique()}")
                # Try converting to the 'Int8' type if possible
                df[col] = df[col].astype('Int8')
                int_cols.append(col)
            except Exception as e:
                df[col] = original_values
                self.logger.error(f"Couldn't simplify {col} to Int because {df[col].unique()} : error {str(e)}")
        self.int_col.extend(int_cols)
        return df, self.int_col

    def int_tran(self, df):
        current_int_col = []  # Track columns converted in this batch

        def int_tran_col(col):
            if np.issubdtype(col.dtype, np.number):
                col_converted = col.round().astype('Int32')
                current_int_col.append(col.name)
                return col_converted
            if col.dtype == 'object':
                conv = pd.to_numeric(col, errors='coerce')
                if col.notna().sum() == conv.notna().sum():
                    current_int_col.append(col.name)
                    return conv.round().astype('Int32')
            return col

        df_transformed = df.apply(int_tran_col)
        self.int_col = current_int_col  # Update class list once
        return df_transformed, self.int_col

    def bool_tran(self, df):
        current_bool_col = []  # Track boolean columns in this batch
        for col_name in df.columns:
            col = df[col_name]
            if pd.api.types.is_numeric_dtype(col):
                unique_vals = col.dropna().unique()
                if len(unique_vals) == 2:
                    sorted_vals = sorted(unique_vals)
                    mapping_bool = {sorted_vals[0]: False, sorted_vals[1]: True}
                    df[col_name] = col.map(mapping_bool)
                    df[col_name] = df[col_name].astype('boolean')
                    current_bool_col.append(col_name)
                    self.logger.info(f"Column '{col_name}': transformed numeric values to boolean {mapping_bool}")

            elif pd.api.types.is_object_dtype(col):
                unique_vals = col.dropna().unique()
                if len(unique_vals) == 2:
                    try:
                        # Create a set of cleaned (lower-case, stripped) unique values
                        uniq_set = set(str(x).strip().lower() for x in unique_vals)
                        # Check if at least one of the unique values starts with "no" or "not"
                        if any(val.startswith("no") or val.startswith("not") or val.startswith('did not') for val in
                               uniq_set):
                            def map_to_bool(x):
                                if pd.isnull(x):
                                    return pd.NA
                                s = str(x).strip().lower()
                                if s.startswith("no") or s.startswith("not") or s.startswith('did not'):
                                    return False
                                else:
                                    return True

                            df[col_name] = col.apply(map_to_bool)
                            df[col_name] = df[col_name].astype('boolean')
                            current_bool_col.append(col_name)
                            self.logger.info(f"Column '{col_name}': transformed object values to boolean.")
                        else:
                            # If neither unique value starts with "no"/"not", skip boolean conversion.
                            self.logger.error(
                                f"Column '{col_name}' has two unique values {unique_vals} but does not match boolean mapping criteria; transformation skipped.")
                    except Exception as e:
                        self.logger.error(
                            f"Couldn't simplify {col_name} to boolean because {df[col_name].unique()} : error {str(e)}")
        self.bool_col = current_bool_col
        return df, self.bool_col

    def drop_empty_columns(self, df):
        empty = []
        for col in df.columns:
            try:
                if df[col].isna().all():
                    empty.append(col)
            except Exception as e:
                empty.append(col)
        self.empty_col_names = list(set(empty))
        df = df.drop(columns=self.empty_col_names, errors="ignore").copy()
        self.logger.info(f"These columns are empty and dropped: {self.empty_col_names}")
        return self.empty_col_names, df

    def student_score_tran(self, df):
        self.relevant_column = [col for col in df.columns
                                if col.endswith('(Scored Response)') or col.endswith('(Coded Response)')]
        df['student note'] = df[self.relevant_column].sum(axis=1, skipna=True, numeric_only=True)
        df['student note'] = df['student note'].astype('int8', errors='ignore')
        df['questions answered'] = df[self.relevant_column].notna().sum(axis=1)
        df['questions answered'] = df['questions answered'].astype('int8', errors='ignore')
        return df, self.relevant_column

    def process_batches(self):
        try:
            self.log("Starting ETL process...")
            _, meta = pyreadstat.read_sav(self.input_file, metadataonly=True)
            total_rows = meta.number_rows
            self.log(f"Total rows in the file: {total_rows}")
            offset = 0

            while offset < total_rows:
                if self.row_limit and self.processed_rows >= self.row_limit:
                    self.log(f"Row limit of {self.row_limit} reached. Stopping ETL process.")
                    break

                try:
                    self.log(
                        f"Processing batch {self.batch_number} (rows {offset} to {min(offset + self.batch_size, total_rows) - 1})...")
                    rows_to_read = min(self.batch_size,
                                       self.row_limit - self.processed_rows) if self.row_limit else self.batch_size
                    df_chunk, meta_chunk = pyreadstat.read_sav(self.input_file, row_offset=offset,
                                                               row_limit=rows_to_read)

                    df_chunk = df_chunk.replace(to_replace=self.nul_vall)
                    df_chunk = df_chunk.replace(to_replace=meta_chunk.variable_value_labels)
                    df_chunk = df_chunk.rename(columns=meta_chunk.column_names_to_labels)

                    if self.batch_number == 1:
                        if self.drop_empty:
                            self.empty_col_names, df_chunk = self.drop_empty_columns(df_chunk)
                        df_chunk, self.int_col = self.int_tran(df_chunk)
                        df_chunk, self.int_col = self.string_tran(df_chunk)
                        df_chunk, self.bool_col = self.bool_tran(df_chunk)
                        if self.student_score:
                            df_chunk, self.relevant_column = self.student_score_tran(df_chunk)

                        self.final_schema_columns = df_chunk.columns.tolist()
                        self.final_dtypes = df_chunk.dtypes.astype(str).to_dict()

                        if self.short_name:
                            self.shortened_mapping = self.generate_shortened_columns(df_chunk)
                            df_chunk = df_chunk.rename(columns=self.shortened_mapping)
                    else:
                        missing_cols = set(self.final_schema_columns) - set(df_chunk.columns)
                        for col in missing_cols:
                            df_chunk[col] = np.nan
                        df_chunk = df_chunk[self.final_schema_columns]
                        df_chunk, self.int_col = self.int_tran(df_chunk)
                        df_chunk, self.int_col = self.string_tran(df_chunk)
                        df_chunk, self.bool_col = self.bool_tran(df_chunk)
                        if self.student_score:
                            df_chunk, self.relevant_column = self.student_score_tran(df_chunk)
                        for col, dtype in self.final_dtypes.items():
                            if dtype == 'Int32':
                                df_chunk[col] = df_chunk[col].astype('float').astype('Int32')
                            elif dtype == 'boolean':
                                df_chunk[col] = df_chunk[col].astype('boolean')
                            else:
                                df_chunk[col] = df_chunk[col].astype(dtype)
                        if self.short_name:
                            df_chunk = df_chunk.rename(columns=self.shortened_mapping)

                    file_name = f"{self.output_dir}/output_file_part{self.batch_number}.parquet"
                    table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                    pq.write_table(table, file_name)

                    self.processed_rows += len(df_chunk)
                    del df_chunk
                    gc.collect()

                except Exception as e:
                    self.log(f"Error reading batch at row {offset}: {e}")
                    break

                offset += self.batch_size
                self.batch_number += 1

            self.log("All batches processed successfully.")

        except Exception as e:
            self.log(f"An error occurred during the ETL process: {e}")
            return None