from loader_module.loader_class import loader
import logging
import pandas as pd
from scipy.stats import norm

df= pd.read_parquet(r'cleaned_out_student_score')

df['z-score question answer'] = (df['questions answered'] - df['questions answered'].mean())/df['questions answered'].std()
df['z-score good answer'] = (df['student note'] - df['student note'].mean())/df['student note'].std()
df['composite_z'] = df['z-score good answer'] - df['z-score question answer']
df['composite_percentile'] = norm.cdf(df['composite_z']) * 100

df.loc[df['student note'] == 0, 'composite_percentile'] = 0

(df[['Country code 3-character','Intl. Student ID','Intl. School ID','questions answered','student note',
    'composite_z', 'z-score good answer','z-score question answer','composite_percentile']]
 .to_parquet('cleaned_out_student_score_normalized.parquet'))


load = loader('cleaned_out_student_score_normalized.parquet', db_name='pisa_final', table_name='student_score_normalized', log_level=logging.INFO)
load.load_data()