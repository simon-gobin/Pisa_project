import os
from ETL_class.ETL_class import ETL

directory = r"C:\Users\simon\Desktop\Pisa\Dataset\questionary"

for entry in os.scandir(directory):
    if entry.is_file() and not entry.name.lower().endswith('.log'):  # check if it's a file and skip log file
        print(entry.path)
        print( os.path.splitext(entry.name)[0])
        etl = ETL(input_file=entry.path, output_dir=f"cleaned_out_{os.path.splitext(entry.name)[0]}", short_name=True, student_score=False)
        etl.process_batches()
    print('No score done')


directory = r"C:\Users\simon\Desktop\Pisa\Dataset\student score"

for entry in os.scandir(directory):
    if entry.is_file() and not entry.name.lower().endswith('.log'):  # check if it's a file and skip log file
        print(entry.path)
        print( os.path.splitext(entry.name)[0])
        etl = ETL(input_file=entry.path, output_dir=f"cleaned_out_{os.path.splitext(entry.name)[0]}", short_name=True, student_score=True)
        etl.process_batches()
    print('Score done')