# CreatorInsight AI/ML Platform  
## End-to-End ML & AI System for YouTube Comment Intelligence

CreatorInsight AI is a production-grade machine learning and AI platform that analyzes YouTube video comments to extract sentiment, audience themes, risks, and actionable insights at scale.

Large creators often receive hundreds or thousands of comments per video, making manual analysis impractical. This system converts unstructured comment data into structured, decision-ready insights using a combination of classical machine learning, modern AI techniques, and MLOps best practices.

Built as a full-stack ML & AI engineering project demonstrating real-world system design, MLOps practices, and cost-optimized AI integration.

---

## Problem Statement

YouTube creators struggle to:
- Analyze large volumes of comments efficiently
- Understand overall audience sentiment
- Identify recurring feedback themes and concerns
- Translate raw feedback into actionable content improvements

Traditional dashboards surface metrics but fail to provide meaningful insights. CreatorInsight AI addresses this gap by combining analytics with intelligent summarization.

---

## Solution Overview

CreatorInsight AI provides:
- Automated sentiment classification of comments
- Audience analytics and visualizations
- Theme discovery from large-scale text data
- Cost-optimized AI-generated summaries
- A Chrome extension interface for one-click analysis

The system is designed to be scalable, explainable, and production-ready.

---

## Key Features

### Sentiment Analysis
- Classifies comments as Positive, Neutral, or Negative using a trained ML model
- Handles noisy user-generated text including slang, emojis, and informal language

### Audience Analytics
- Sentiment distribution and trend analysis
- Engagement metrics such as average comment length and unique commenters

### Theme Discovery
- Uses semantic embeddings and clustering to group similar comments
- Identifies dominant discussion topics across large comment sets

### AI Comment Summarization
- Generates structured insights including:
  - Key discussion themes
  - What the audience liked
  - Risks and concerns
  - Actionable content suggestions
- Uses a cost-aware summarization pipeline to minimize LLM usage

### Reporting
- Exports insights as a clean, browser-rendered HTML report
- Suitable for sharing, printing, or saving as PDF

---

## AI & ML Architecture

The system combines classical ML and modern AI techniques:

1. Comment collection via YouTube Data API  
2. Text preprocessing and noise reduction  
3. Sentiment classification using classical ML  
4. Semantic embeddings for comment representation  
5. Clustering to identify discussion themes  
6. Representative sampling to reduce input size  
7. LLM-based map-reduce summarization  
8. Structured JSON output for UI rendering  

This architecture significantly reduces LLM token usage while maintaining summary quality.

---

## System Architecture

work in progress ...