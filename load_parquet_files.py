import os
from loader_module.loader_class import loader
import logging

directory = r'C:\Users\simon\PycharmProjects\Pisa ETL + AIML'

prefix = "cleaned_out_"

for entry in os.scandir(directory):
    if entry.name.lower().startswith(prefix):
        table_name = entry.name[len(prefix):]
        print(entry.path)
        print(table_name)
        load = loader(entry.path, db_name = 'pisa_final', table_name= table_name, log_level=logging.INFO)
        load.load_data()

print('load done')