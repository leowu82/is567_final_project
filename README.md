# Multi-Layered Fraud Detection System

## Introduction & Motivation
As cyber threats evolve, traditional keyword-based spam filters are becoming outdated, particularly against highly targeted phishing and social engineering attacks. Modern fraud often lacks obvious malicious payloads (like malware links), relying instead on psychological manipulation and authority spoofing. This project proposes a multi-layered fraud detection system that moves beyond deterministic rule sets. By combining traditional Natural Language Processing (NLP) techniques with advanced Small Language Models (SLMs) and Retrieval-Augmented Generation (RAG), this system aims to accurately classify text communications (emails and SMS) and provide natural language explainability for its decisions.

## Project Objectives
1. **Baseline vs. Deep Learning Comparison:** Compare the effectiveness of traditional sparse vectorizations (TF-IDF + classic ML) against dense contextual embeddings (Word2Vec/Transformers) in identifying fraudulent language.
2. **Semantic Evaluation:** Train an SLM to perform binary classification: Safe vs Suspicious.
3. **Contextual Awareness via RAG:** Implement a RAG pipeline utilizing open-source threat databases to detect zero-day or rapidly evolving scam templates.
4. **Explainability:** Generate natural language summaries detailing why a text was flagged, fostering trust and transparency in the model's output.

## System Architecture & Methodology

### Phase 1: The Gatekeeper (Preprocessing & Heuristics)
This initial layer cleans the data and performs rapid, low-compute filtering.
* **PII Scrubbing:** Redaction of sensitive information (names, numbers) to ensure data privacy before processing.
* **Rule-Based Extraction (Regex):** Utilizing Regular Expressions to capture rigid, known patterns like standard phishing URLs, crypto wallet addresses, and routing numbers.
* **Statistical Representation:** We will use TF-IDF paired with a traditional classifier (like SVM) to establish our baseline model.

### Phase 2: The Deep Analysis (Semantic Layer & RAG)
Communications that bypass the Gatekeeper are routed to this computationally heavier layer.
* **SLM Classification:** Fine-tuning a Small Language Model to evaluate the overall semantic payload, categorizing the text as Safe or Suspicious.
* **Retrieval-Augmented Generation (RAG):** Text will be embedded and queried against a Vector Database of known, current scams. 
* **Sources:** The database will be populated using the FTC Open Data API, archival database dumps of the r/Scams subreddit, and Phish Tank's verified phishing URLs.

### Phase 3: Output & Explainability
* **Natural Language Summarization:** If a text is flagged as Suspicious, the system will output a generated summary explaining the threat vector, such as highlighting high urgency tone or similarity to a known scam template in the FTC database.

## Datasets
* **Phishing Email Dataset:** Contains ~82,500 legitimate and phishing emails. This dataset is well-balanced.
* **SMS Spam Collection Dataset:** Contains 5,574 SMS messages tagged as spam or ham. It suffers from a severe class imbalance, sitting at about 87% legitimate messages to 13% spam.
* **RAG Knowledge Base:** Data ingested via the FTC Open Data API, PhishTank's developer database, and archival, open-source datasets of r/Scams.
* **Challenges:** Cross-referencing datasets against the Stanford framework revealed gaps in specific categories of fraud. In the future we plan to use an LLM to generate synthetic data to fill these categories.

## Evaluation Metrics
The system's performance will be quantified using a Confusion Matrix to track multi-class performance (Safe vs. Suspicious vs. Inconclusive), alongside standard Accuracy, Precision, Recall, and F1-Scores. 
**Recall is prioritized** as the primary metric, as minimizing False Negatives (allowing a scam to reach a user) is strictly more important than minimizing False Positives in a security context.
