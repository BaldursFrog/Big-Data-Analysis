# Financial Sentiment Analysis: BERT vs FinBERT

This laboratory work focuses on a comparative study of the general-purpose language model **BERT** and the domain-specific model **FinBERT** in the task of financial sentiment analysis.

## Objective
To analyze the impact of domain adaptation in language models and experimentally compare the performance of BERT and FinBERT on financial text sentiment classification.

## Tasks
- Study the core mechanisms of BERT (bidirectional encoding, MLM, NSP)
- Analyze domain-specific improvements introduced in FinBERT
- Conduct an experimental comparison of BERT and FinBERT
- Evaluate model performance using accuracy, precision, recall, and F1-score
- Visualize and interpret experimental results

## Experiment Description
The experiment uses a synthetic dataset of 1000 financial text samples with three sentiment classes:
- Positive (40%)
- Neutral (35%)
- Negative (25%)

For each model, the following metrics are computed:
- Accuracy
- Precision
- Recall
- F1-score
- Average confidence score

Comparative visualizations are generated based on these metrics.

## Project Structure
├── experimental_comparison.py # Experiment logic and visualization code
├── bert_finbert_comparison.png # Model performance comparison
├── confidence_by_sentiment.png # Confidence scores by sentiment
├── README.md # Project description


## Technologies Used
- Python 3.9+
- NumPy
- Matplotlib
- Pretrained NLP models (BERT, FinBERT – conceptually)

## Results
The experimental results demonstrate that FinBERT consistently outperforms BERT across all evaluation metrics:
- Higher classification accuracy
- Greater prediction confidence
- Better differentiation between financial sentiment classes

These results highlight the advantages of domain-specific pretraining for financial text analysis.

## Conclusion
Domain-adapted language models such as FinBERT are more effective for financial sentiment analysis tasks, as they better capture domain-specific vocabulary and contextual nuances compared to general-purpose models.
