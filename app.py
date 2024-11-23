from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from config import Config
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import folium
import json

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize PySpark
spark = SparkSession.builder \
    .appName("Task 6: Delayed Flights Analysis") \
    .master("local[*]") \
    .getOrCreate()


# -------------------------- Routes --------------------------

# Dashboard
@app.route("/")
def dashboard():
    """Render the main dashboard."""
    return render_template("dashboard.html")


# Task 1: Delayed Flights
@app.route("/task1")
def task1():
    try:
        # Define the delay-related pattern
        delay_pattern = r"(?i)(delay|delayed|waiting|postponed|canceled|late|rescheduled|held|deferred)"

        # Query to count delays by airline
        airline_query = text(f"""
            SELECT airline, COUNT(*) AS count
            FROM airline_sentiments
            WHERE text ~ '{delay_pattern}'
            GROUP BY airline
            ORDER BY count DESC;
        """)
        airline_data = db.session.execute(airline_query).fetchall()

        # Query to count delays by location
        location_query = text(f"""
            SELECT _country AS location, COUNT(*) AS count
            FROM airline_sentiments
            WHERE text ~ '{delay_pattern}'
            GROUP BY _country
            ORDER BY count DESC;
        """)
        location_data = db.session.execute(location_query).fetchall()

        # Total delayed mentions
        total_delays = sum(row[1] for row in airline_data)

        return render_template(
            "task1.html",
            title="Task 1: Delayed Flights",
            total_delays=total_delays,
            airline_data=airline_data,
            location_data=location_data
        )
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

# Task 2: Top Negative Reasons
@app.route("/task2")
def task2():
    query = text("""
        WITH ranked_reasons AS (
            SELECT 
                airline, 
                negativereason, 
                COUNT(*) AS reason_count,
                RANK() OVER (PARTITION BY airline ORDER BY COUNT(*) DESC) AS rank
            FROM airline_sentiments
            WHERE airline_sentiment = 'negative'
            AND negativereason IS NOT NULL
            GROUP BY airline, negativereason
        )
        SELECT airline, negativereason, reason_count
        FROM ranked_reasons
        WHERE rank <= 5
        ORDER BY airline, rank;
    """)
    results = db.session.execute(query).fetchall()
    
    # Format results to pass to the template
    formatted_data = [{"airline": row[0], "negativereason": row[1], "count": row[2]} for row in results]

    return render_template(
        "task2.html",
        title="Task 2: Top Negative Reasons",
        data=formatted_data,
        columns=["Airline", "Negative Reason", "Count"]
    )




# Task 3: Country-wise Complaints
@app.route("/task3")
def task3():
    query = text("""
    WITH country_data AS (
        SELECT
            TRIM(_country) AS country,
            COUNT(*) AS complaint_count,
            SUM(COUNT(*)) OVER () AS total_complaints
        FROM airline_sentiments
        WHERE airline_sentiment = 'negative'
            AND _country IS NOT NULL
            AND _country NOT IN ('unknown', 'country')
        GROUP BY _country
    ),
    reason_data AS (
        SELECT
            _country AS country,
            negativereason AS top_complaint_reason
        FROM (
            SELECT
                _country,
                negativereason,
                ROW_NUMBER() OVER (PARTITION BY _country ORDER BY COUNT(*) DESC) AS reason_rank
            FROM airline_sentiments
            WHERE airline_sentiment = 'negative'
                AND _country IS NOT NULL
                AND _country != 'unknown'
            GROUP BY _country, negativereason
        ) ranked_reasons
        WHERE reason_rank = 1
    )
    SELECT
        c.country,
        c.complaint_count,
        ROUND((c.complaint_count * 100.0) / c.total_complaints, 2) AS percentage_of_total,
        COALESCE(r.top_complaint_reason, 'N/A') AS top_complaint_reason
    FROM country_data c
    LEFT JOIN reason_data r
    ON c.country = r.country
    ORDER BY c.complaint_count DESC;
""")


    results = db.session.execute(query).fetchall()

    # Transforming the result for rendering
    data = [
        {
            "Country": row[0],
            "Complaint Count": row[1],
            "Percentage of Total (%)": row[2],
            "Top Complaint Reason": row[3],
        }
        for row in results
    ]

    return render_template(
        "task3.html",
        title="Task 3: Country-wise Complaints",
        data=data,
        columns=["Country", "Complaint Count", "Percentage of Total (%)", "Top Complaint Reason"]
    )


# Task 4: Trust Scores
# Task 4: Trust Analysis
@app.route("/task4")
def task4():
    try:
        # SQL Query to calculate mean and median trust scores for each channel
        query = text("""
            WITH trust_data AS (
                SELECT
                    _channel AS channel,
                    _trust AS trust_score
                FROM airline_sentiments
                WHERE _channel IS NOT NULL AND _trust IS NOT NULL
            )
            SELECT
                channel,
                ROUND(AVG(trust_score)::numeric, 2) AS mean_trust_score,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trust_score)::numeric, 2) AS median_trust_score
            FROM trust_data
            GROUP BY channel
            ORDER BY channel;
        """)

        # Execute the query
        results = db.session.execute(query).fetchall()

        # Prepare data for rendering
        data = [{"channel": row[0], "mean_trust_score": row[1], "median_trust_score": row[2]} for row in results]

        return render_template(
            "task4.html",
            title="Task 4: Trust Analysis",
            data=data,
            columns=["Channel", "Mean Trust Score", "Median Trust Score"]
        )
    except Exception as e:
        return f"Error occurred: {str(e)}", 500



# Task 5: Sentiment Analysis Results
@app.route("/task5")
def task5():
    query = text("""
        SELECT airline_sentiment, COUNT(*) AS sentiment_count
        FROM airline_sentiments
        GROUP BY airline_sentiment
        ORDER BY sentiment_count DESC;
    """)
    results = db.session.execute(query).fetchall()
    return render_template("task.html", title="Task 5: Sentiment Analysis", data=results, columns=["Sentiment", "Count"])


# Task 6: Dynamic Visualizations
@app.route("/task6")
def task6():
    try:
        # Sentiment distribution by airline
        sentiment_query = text("""
            SELECT airline, airline_sentiment, COUNT(*) AS count
            FROM airline_sentiments
            GROUP BY airline, airline_sentiment
            ORDER BY count DESC;
        """)
        sentiment_data = db.session.execute(sentiment_query).fetchall()

        sentiment_json = [
            {"airline": row[0], "sentiment": row[1], "count": row[2]} for row in sentiment_data
        ]

        # Average trust score by airline
        trust_query = text("""
            SELECT airline, AVG(_trust) AS avg_trust
            FROM airline_sentiments
            GROUP BY airline
            ORDER BY avg_trust DESC;
        """)
        trust_data = db.session.execute(trust_query).fetchall()

        avg_trust_json = [{"airline": row[0], "avg_trust": row[1]} for row in trust_data]

        # Country-wise complaints
        country_complaints_query = text("""
            SELECT _country, COUNT(*) AS complaints_count
            FROM airline_sentiments
            GROUP BY _country
            ORDER BY complaints_count DESC
            LIMIT 10;
        """)
        country_data = db.session.execute(country_complaints_query).fetchall()

        country_json = [{"country": row[0], "count": row[1]} for row in country_data]

        # Positive sentiment distribution
        positive_sentiment_query = text("""
            SELECT airline, COUNT(*) AS count
            FROM airline_sentiments
            WHERE airline_sentiment = 'positive'
            GROUP BY airline
            ORDER BY count DESC;
        """)
        positive_sentiment_data = db.session.execute(positive_sentiment_query).fetchall()

        positive_sentiment_json = [{"airline": row[0], "count": row[1]} for row in positive_sentiment_data]

        return render_template(
            "task6.html",
            title="Task 6: Detailed Analysis",
            sentiment_json=sentiment_json,
            avg_trust_json=avg_trust_json,
            country_json=country_json,
            positive_sentiment_json=positive_sentiment_json,
        )

    except Exception as e:
        return f"Error occurred: {str(e)}", 500



# -------------------------- Run App --------------------------
if __name__ == "__main__":
    app.run(debug=True)