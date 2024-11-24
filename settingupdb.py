import pandas as pd
from sqlalchemy import create_engine, text

# File path for the CSV file
file_path = "Airline-Full-Non-Ag-DFE-Sentiment.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Clean column names (optional)
df.columns = df.columns.str.strip().str.lower()

# Connect to PostgreSQL
engine = create_engine('postgresql://postgres:cloudproject2group6@cloudproject2.cjwb4bnnxrdb.us-east-1.rds.amazonaws.com:5432/postgres')

# SQL script to create the table with the specified name
create_table_sql = """
CREATE TABLE IF NOT EXISTS "Airline-Full-Non-Ag-DFE-Sentiment" (
    golden BOOLEAN,
    channel VARCHAR(50),
    trust NUMERIC,
    country VARCHAR(3),
    airline_sentiment VARCHAR(50),
    negativereason VARCHAR(100),
    airline VARCHAR(50),
    sentiment_gold VARCHAR(50),
    negativereason_gold VARCHAR(100),
    text TEXT
);
"""

# Create the table in PostgreSQL
with engine.connect() as connection:
    connection.execute(text(create_table_sql))
    print("Table 'Airline-Full-Non-Ag-DFE-Sentiment' created successfully.")

# Write the DataFrame to PostgreSQL
df.to_sql('Airline-Full-Non-Ag-DFE-Sentiment', engine, if_exists='append', index=False)
print("Data imported successfully!")
