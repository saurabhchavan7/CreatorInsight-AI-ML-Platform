# app.py

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot

import json

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
import matplotlib.dates as mdates
import os


import boto3
from datetime import datetime

# NLTK imports (safe-guarded usage below)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# AI summarization imports
# OpenAI Configuration
# -----------------------------
# OpenAI Configuration (matches MLflow pattern)
# -----------------------------
from openai import OpenAI
import os

try:
    from config import OPENAI_API_KEY as CONFIG_OPENAI_KEY
except ImportError:
    CONFIG_OPENAI_KEY = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or CONFIG_OPENAI_KEY

# Simple initialization - no extra arguments
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("[INFO] OpenAI client initialized")
    except Exception as e:
        print(f"[ERROR] OpenAI failed: {e}")
        openai_client = None
else:
    print("[WARNING] OPENAI_API_KEY not set")



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "http://ec2-3-91-67-139.compute-1.amazonaws.com:5000/"  # ← FALLBACK HERE


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# print(f"[INFO] Using MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

MODEL_NAME = "creatorinsight_sentiment_pipeline"
MODEL_VERSION = "Production"  # you can switch to "Production" later if you use stages



# -----------------------------
# AWS CloudWatch Configuration
# -----------------------------

import time
from functools import wraps

# CloudWatch logging
try:
    import boto3
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    CLOUDWATCH_ENABLED = True
    print("[INFO] CloudWatch metrics enabled")
except Exception as e:
    CLOUDWATCH_ENABLED = False
    print(f"[WARNING] CloudWatch disabled: {e}")

# -----------------------------
# AWS CloudWatch Decorator
# -----------------------------

def monitor_endpoint(metric_name):
    """Decorator to monitor API endpoints with CloudWatch"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error_occurred = False
            
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                raise
            finally:
                # Calculate response time
                response_time = (time.time() - start_time) * 1000  # milliseconds
                
                # Log to CloudWatch
                if CLOUDWATCH_ENABLED:
                    try:
                        cloudwatch.put_metric_data(
                            Namespace='CreatorInsight/API',
                            MetricData=[
                                {
                                    'MetricName': f'{metric_name}_ResponseTime',
                                    'Value': response_time,
                                    'Unit': 'Milliseconds',
                                    'Timestamp': time.time()
                                },
                                {
                                    'MetricName': f'{metric_name}_RequestCount',
                                    'Value': 1,
                                    'Unit': 'Count',
                                    'Timestamp': time.time()
                                },
                                {
                                    'MetricName': f'{metric_name}_ErrorCount',
                                    'Value': 1 if error_occurred else 0,
                                    'Unit': 'Count',
                                    'Timestamp': time.time()
                                }
                            ]
                        )
                    except Exception as cw_error:
                        print(f"CloudWatch logging failed: {cw_error}")
        
        return wrapper
    return decorator


# -----------------------------
# Text Preprocessing (safe)
# -----------------------------
# Define the preprocessing function
def preprocess_comment(comment) -> str:
    """
    Safe preprocessing:
    - Handles int, None, float, etc.
    - Never crashes the pipeline
    """
    try:
        # Convert everything to string safely
        if comment is None:
            return ""

        comment = str(comment)

        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        try:
            sw = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
        except Exception:
            sw = set()

        words = [w for w in comment.split() if w not in sw]
        comment = " ".join(words)

        try:
            lemmatizer = WordNetLemmatizer()
            comment = " ".join(lemmatizer.lemmatize(word) for word in comment.split())
        except Exception:
            pass

        return comment

    except Exception as e:
        print(f"Preprocessing failed for comment: {comment}, error: {e}")
        return str(comment)



# -----------------------------
# Load MLflow pipeline model (ONE artifact)
# -----------------------------
def load_pipeline_from_registry(model_name: str, model_version: str):
    """
    Loads a single MLflow-registered pipeline model (TF-IDF + LGBM).
    This replaces separate loading of vectorizer + model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{model_version}"
    return mlflow.sklearn.load_model(model_uri)


# Load once at startup
try:
    pipeline = load_pipeline_from_registry(MODEL_NAME, MODEL_VERSION)
    print(f"Loaded MLflow pipeline model: {MODEL_NAME} v{MODEL_VERSION}")
except Exception as e:
    pipeline = None
    print(f"Failed to load MLflow pipeline model: {e}")


def ensure_pipeline_loaded():
    """Fail fast if model isn't loaded (better error for debugging)."""
    if pipeline is None:
        return False, jsonify({"error": "Model pipeline not loaded. Check MLflow URI/model name/version."}), 500
    return True, None, None



# -----------------------------
# AI Summary Generation
# -----------------------------


def generate_ai_summary(comments, sentiments):
    """
    Generate AI summary using OpenAI GPT-4o-mini
    
    Args:
        comments: List of comment texts (max 500)
        sentiments: List of predictions (-1, 0, 1)
    
    Returns:
        dict with key_themes, audience_loved, risks_concerns, suggestions
    """
    if not openai_client:
        return {
            "error": "OpenAI API not configured",
            "key_themes": "API key missing",
            "what_audience_loved": "Configure OPENAI_API_KEY",
            "risks_concerns": "N/A",
            "actionable_suggestions": "Set up OpenAI API key"
        }
    
    try:
        # Group comments by sentiment
        positive_comments = [comments[i] for i in range(len(comments)) if str(sentiments[i]) == '1']
        negative_comments = [comments[i] for i in range(len(comments)) if str(sentiments[i]) == '-1']
        neutral_comments = [comments[i] for i in range(len(comments)) if str(sentiments[i]) == '0']
        
        # Limit to prevent token overflow (GPT-4o-mini has 128k context, but we'll be safe)
        def sample_comments(comment_list, max_count=100):
            """Take first and last comments to get variety"""
            if len(comment_list) <= max_count:
                return comment_list
            half = max_count // 2
            return comment_list[:half] + comment_list[-half:]
        
        pos_sample = sample_comments(positive_comments, 100)
        neg_sample = sample_comments(negative_comments, 100)
        neu_sample = sample_comments(neutral_comments, 50)
        
        # Calculate percentages for context
        total = len(sentiments)
        pos_pct = (len(positive_comments) / total * 100) if total > 0 else 0
        neg_pct = (len(negative_comments) / total * 100) if total > 0 else 0
        neu_pct = (len(neutral_comments) / total * 100) if total > 0 else 0
        
        # Build prompt
        prompt = f"""You are analyzing YouTube comments for a content creator. Here's the data:

**Statistics:**
- Total comments: {total}
- Positive: {len(positive_comments)} ({pos_pct:.1f}%)
- Negative: {len(negative_comments)} ({neg_pct:.1f}%)
- Neutral: {len(neutral_comments)} ({neu_pct:.1f}%)

**Sample Positive Comments:**
{chr(10).join(['- ' + c[:200] for c in pos_sample[:20]])}

**Sample Negative Comments:**
{chr(10).join(['- ' + c[:200] for c in neg_sample[:20]])}

**Sample Neutral Comments:**
{chr(10).join(['- ' + c[:200] for c in neu_sample[:10]])}

Provide a concise creator-friendly analysis in this EXACT JSON format:
{{
  "key_themes": "2-3 sentence summary of main topics discussed",
  "what_audience_loved": "What viewers appreciated (be specific, reference actual topics)",
  "risks_concerns": "What viewers didn't like or complained about (be specific)",
  "actionable_suggestions": "2-3 concrete actions the creator should take"
}}

Be specific and reference actual topics from comments. Keep each field under 150 words."""

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Cheapest, fastest
            messages=[
                {"role": "system", "content": "You are a YouTube analytics expert helping creators improve their content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600,  # Keep response concise
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        # Parse response
        result = json.loads(response.choices[0].message.content)
        
        # Add metrics
        result["metrics"] = {
            "total_comments": total,
            "positive_percentage": round(pos_pct, 1),
            "neutral_percentage": round(neu_pct, 1),
            "negative_percentage": round(neg_pct, 1),
            "sentiment_score": round((sum([int(s) for s in sentiments]) / total + 1) * 5, 1)
        }
        
        return result
        
    except Exception as e:
        app.logger.error(f"OpenAI API error: {e}")
        return {
            "error": str(e),
            "key_themes": f"Error: {str(e)}",
            "what_audience_loved": "Analysis failed",
            "risks_concerns": "Check logs",
            "actionable_suggestions": "Retry analysis"
        }
    



# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return "Welcome to our flask api"


# @app.route("/predict_with_timestamps", methods=["POST"])
# @monitor_endpoint("PredictWithTimestamps")  # ADD THIS LINE for monitoring on this endpoint at CloudWatch
# def predict_with_timestamps():
#     ok, resp, code = ensure_pipeline_loaded()
#     if not ok:
#         return resp, code

#     data = request.json
#     comments_data = data.get("comments")

#     if not comments_data:
#         return jsonify({"error": "No comments provided"}), 400

#     try:
#         comments = [item.get("text", "") for item in comments_data]
#         timestamps = [item.get("timestamp") for item in comments_data]

#         preprocessed_comments = [preprocess_comment(comment) for comment in comments]

#         # Get predictions
#         preds = pipeline.predict(preprocessed_comments)
        
#         # Get confidence scores (NEW!)
#         try:
#             proba = pipeline.predict_proba(preprocessed_comments)
#             # Confidence = max probability for predicted class
#             confidences = [float(max(prob)) for prob in proba]
#         except Exception as e:
#             print(f"Warning: Could not get confidence scores: {e}")
#             confidences = [None] * len(preds)

#         # Normalize predictions
#         if isinstance(preds, (np.ndarray, list)):
#             predictions = [str(p) for p in list(preds)]
#         else:
#             predictions = [str(preds)]

        
#         # ========================================
#         # ADD CLOUDWATCH LOGGING HERE 
#         # ========================================
#         if CLOUDWATCH_ENABLED and confidences:
#             avg_confidence = sum(c for c in confidences if c is not None) / len(confidences)
            
#             cloudwatch.put_metric_data(
#                 Namespace='CreatorInsight/Model',
#                 MetricData=[
#                     {
#                         'MetricName': 'AverageConfidence',
#                         'Value': avg_confidence,
#                         'Unit': 'None',
#                         'Timestamp': time.time()
#                     },
#                     {
#                         'MetricName': 'PredictionCount',
#                         'Value': len(predictions),
#                         'Unit': 'Count',
#                         'Timestamp': time.time()
#                     }
#                 ]
#             )
#         # ========================================
#         # END OF CLOUDWATCH LOGGING
#         # ========================================


#     except Exception as e:
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

#     response = [
#         {
#             "comment": comment, 
#             "sentiment": sentiment, 
#             "timestamp": timestamp,
#             "confidence": confidence  # NEW FIELD!
#         }
#         for comment, sentiment, timestamp, confidence 
#         in zip(comments, predictions, timestamps, confidences)
#     ]
#     return jsonify(response)


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    start_time = time.time()
    error_occurred = False
    
    ok, resp, code = ensure_pipeline_loaded()
    if not ok:
        return resp, code

    data = request.json
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item.get("text", "") for item in comments_data]
        timestamps = [item.get("timestamp") for item in comments_data]
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Get predictions and confidence
        preds = pipeline.predict(preprocessed_comments)
        
        try:
            proba = pipeline.predict_proba(preprocessed_comments)
            confidences = [float(max(prob)) for prob in proba]
            avg_confidence = sum(confidences) / len(confidences)
        except Exception as e:
            print(f"Warning: Could not get confidence scores: {e}")
            confidences = [None] * len(preds)
            avg_confidence = None

        if isinstance(preds, (np.ndarray, list)):
            predictions = [str(p) for p in list(preds)]
        else:
            predictions = [str(preds)]
        
        # CloudWatch logging
        response_time = (time.time() - start_time) * 1000
        
        if CLOUDWATCH_ENABLED:
            try:
                metrics = [
                    {
                        'MetricName': 'API_ResponseTime',
                        'Value': response_time,
                        'Unit': 'Milliseconds'
                    },
                    {
                        'MetricName': 'API_RequestCount',
                        'Value': 1,
                        'Unit': 'Count'
                    },
                    {
                        'MetricName': 'PredictionCount',
                        'Value': len(predictions),
                        'Unit': 'Count'
                    }
                ]
                
                if avg_confidence:
                    metrics.append({
                        'MetricName': 'AverageConfidence',
                        'Value': avg_confidence,
                        'Unit': 'None'
                    })
                
                cloudwatch.put_metric_data(
                    Namespace='CreatorInsight',
                    MetricData=metrics
                )
            except Exception as e:
                print(f"CloudWatch error: {e}")

    except Exception as e:
        error_occurred = True
        
        if CLOUDWATCH_ENABLED:
            cloudwatch.put_metric_data(
                Namespace='CreatorInsight',
                MetricData=[{
                    'MetricName': 'API_ErrorCount',
                    'Value': 1,
                    'Unit': 'Count'
                }]
            )
        
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {
            "comment": comment,
            "sentiment": sentiment,
            "timestamp": timestamp,
            "confidence": confidence
        }
        for comment, sentiment, timestamp, confidence
        in zip(comments, predictions, timestamps, confidences)
    ]
    return jsonify(response)


@app.route("/predict", methods=["POST"])
@monitor_endpoint("Predict")  # ADD
def predict():
    ok, resp, code = ensure_pipeline_loaded()
    if not ok:
        return resp, code

    data = request.json
    comments = data.get("comments")

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        preds = pipeline.predict(preprocessed_comments)

        # Get confidence scores (NEW!)
        try:
            proba = pipeline.predict_proba(preprocessed_comments)
            # Confidence = max probability for predicted class
            confidences = [float(max(prob)) for prob in proba]
        except Exception as e:
            print(f"Warning: Could not get confidence scores: {e}")
            confidences = [None] * len(preds)

        if isinstance(preds, (np.ndarray, list)):
            predictions = [str(p) for p in list(preds)]
        else:
            predictions = [str(preds)]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment, "confidence": confidence} for comment, sentiment, confidence in zip(comments, predictions, confidences)]
    return jsonify(response)


@app.route("/generate_chart", methods=["POST"])
@monitor_endpoint("GenerateChart")  # ADD
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#2F6FED", "#9AA4B2", "#E5484D"]  # subtle + premium

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "#111827"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route("/generate_wordcloud", methods=["POST"])
@monitor_endpoint("GenerateWordcloud")  # ADD
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get("comments")

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = " ".join(preprocessed_comments)

        # Safe stopwords
        try:
            sw = set(stopwords.words("english"))
        except Exception:
            sw = set()

        wordcloud = WordCloud(
            width=900,
            height=450,
            background_color="white",
            colormap="Blues",
            stopwords=sw,
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route("/generate_trend_graph", methods=["POST"])
@monitor_endpoint("GenerateTrendGraph")  # ADD
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data")

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))

        colors = {-1: "#E5484D", 0: "#9AA4B2", 1: "#2F6FED"}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker="o",
                linestyle="-",
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500
    
@app.route("/generate_ai_summary", methods=["POST"])
@monitor_endpoint("SaveForRetraining")  # ADD
def generate_ai_summary_endpoint():
    """
    Generate AI-powered summary from comments
    
    Request body:
    {
      "comments": ["comment1", "comment2", ...],
      "sentiments": ["1", "-1", "0", ...]  # Optional, will predict if not provided
    }
    """
    ok, resp, code = ensure_pipeline_loaded()
    if not ok:
        return resp, code
    
    try:
        data = request.get_json()
        comments = data.get("comments", [])
        sentiments = data.get("sentiments")
        
        if not comments:
            return jsonify({"error": "No comments provided"}), 400
        
        # Limit to 500 comments
        comments = comments[:500]
        
        # If sentiments not provided, predict them
        if not sentiments:
            preprocessed = [preprocess_comment(c) for c in comments]
            predictions = pipeline.predict(preprocessed)
            sentiments = [str(p) for p in predictions]
        else:
            sentiments = sentiments[:500]  # Match comment limit
        
        # Generate AI summary
        summary = generate_ai_summary(comments, sentiments)
        
        return jsonify(summary), 200
        
    except Exception as e:
        app.logger.error(f"Error in /generate_ai_summary: {e}")
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500


@app.route("/save_for_retraining", methods=["POST"])
def save_for_retraining():
    """
    Save predictions to S3 for future retraining
    
    Request body:
    {
      "video_id": "dQw4w9WgXcQ",
      "comments": ["comment1", "comment2", ...],
      "predictions": [
        {"comment": "...", "sentiment": "1", "confidence": 0.95, "timestamp": "..."},
        ...
      ]
    }
    """
    try:
        data = request.get_json()
        video_id = data.get("video_id", "unknown")
        predictions = data.get("predictions", [])
        
        if not predictions or len(predictions) == 0:
            return jsonify({"error": "No predictions to save"}), 400
        
        # Create save data with metadata
        save_data = {
            "video_id": video_id,
            "saved_at": datetime.now().isoformat(),
            "count": len(predictions),
            "metadata": {
                "avg_confidence": sum(p.get("confidence", 0) for p in predictions if p.get("confidence")) / len(predictions),
                "sentiment_distribution": {
                    "positive": sum(1 for p in predictions if str(p.get("sentiment")) == "1"),
                    "neutral": sum(1 for p in predictions if str(p.get("sentiment")) == "0"),
                    "negative": sum(1 for p in predictions if str(p.get("sentiment")) == "-1")
                }
            },
            "data": predictions
        }
        
        # Save to S3
        s3 = boto3.client('s3')
        bucket = "creator-insight-dvc-bucket"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"retraining_data/{timestamp}_{video_id}.json"
        
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(save_data, indent=2),
            ContentType='application/json'
        )
        
        print(f"Saved {len(predictions)} predictions to S3: {key}")
        
        return jsonify({
            "status": "success",
            "count": len(predictions),
            "s3_key": key,
            "avg_confidence": save_data["metadata"]["avg_confidence"]
        }), 200
        
    except Exception as e:
        app.logger.error(f"Save to S3 failed: {e}")
        return jsonify({"error": f"Save failed: {str(e)}"}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
