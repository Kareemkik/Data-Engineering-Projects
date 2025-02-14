from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from _functions.cleaning import extract_data, extract_states, combine_sources, encoding , load_to_db

# Define the DAG
default_args = {
    "owner": "data_engineering_team",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 0,
}

with DAG(
    dag_id = 'fintech_dag',
    schedule_interval = '@once', # could be @daily, @hourly, etc or a cron expression '* * * * *'
    default_args = default_args,
    tags = ['pipeline', 'etl', 'sales'],
)as dag:
    # Define the tasks
    extract_fintech = PythonOperator(
        task_id = 'extract_fintech',
        python_callable = extract_data,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_data_1_62_01554.csv',
            'output_path': '/opt/airflow/data/df_fintech.parquet'
        }
    )

    extract_states_data = PythonOperator(
        task_id = 'extract_states',
        python_callable = extract_states,
        op_kwargs = {
            'filename': '/opt/airflow/data/states.csv',
            'output_path': '/opt/airflow/data/states_extract.parquet'
        }
    )

    df_fintech_Combined=PythonOperator(
        task_id = 'df_fintech_Combined',
        python_callable = combine_sources,
        op_kwargs = {
            'filename': '/opt/airflow/data/df_fintech.parquet',
            'filename1':'/opt/airflow/data/states_extract.parquet',
            'output_path': '/opt/airflow/data/df_combined.parquet'
        }

    )

    df_encoded=PythonOperator(
        task_id = 'df_encoded',
        python_callable = encoding,
        op_kwargs = {
            'filename': '/opt/airflow/data/df_combined.parquet',
            'output_path': '/opt/airflow/data/df_encoded.parquet'
        }


    )

    load_to_postgres = PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_db,
        op_kwargs = {
            'filename': '/opt/airflow/data/df_encoded.parquet',
            'table_name': 'fintech_clean',
            'postgres_opt': {
                'user': 'root',
                'password': 'root',
                'host': 'pgdatabase',
                'port': 5432,
                'db': 'data_engineering'
            }
        }
    )

    # Define the task dependencies
    extract_states_data >> df_fintech_Combined >> df_encoded >> load_to_postgres
    extract_fintech  >> df_fintech_Combined >> df_encoded >> load_to_postgres