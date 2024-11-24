from sqlalchemy import create_engine, text

# Connect to Amazon RDS PostgreSQL instance
engine = create_engine('postgresql://postgres:cloudproject2group6@cloudproject2.cjwb4bnnxrdb.us-east-1.rds.amazonaws.com:5432/postgres')

# SQL to rename the table
rename_table_sql = """
ALTER TABLE "Airline-Full-Non-Ag-DFE-Sentiment" RENAME TO airline_sentiments;
"""

try:
    with engine.connect() as connection:
        # Execute the ALTER TABLE command
        connection.execute(text(rename_table_sql))
        print("Table renamed successfully to 'airline_sentiments'.")
except Exception as e:
    print(f"An error occurred: {e}")
