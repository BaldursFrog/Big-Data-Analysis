import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_synthetic_data(n_samples=1000):
    sentiments = ['positive', 'neutral', 'negative']
    sentiment_weights = [0.4, 0.35, 0.25]
    
    true_sentiments = np.random.choice(sentiments, size=n_samples, p=sentiment_weights)
    
    bert_predictions = []
    bert_confidence = []
    
    for sentiment in true_sentiments:
        if sentiment == 'positive':
            pred_probs = [0.75, 0.15, 0.10]
        elif sentiment == 'neutral':
            pred_probs = [0.20, 0.65, 0.15]
        else:
            pred_probs = [0.15, 0.20, 0.65]
        
        noise = np.random.normal(0, 0.05, 3)
        pred_probs = np.array(pred_probs) + noise
        pred_probs = np.clip(pred_probs, 0, 1)
        pred_probs = pred_probs / pred_probs.sum()
        
        pred_sentiment = np.random.choice(sentiments, p=pred_probs)
        confidence = pred_probs.max()
        
        bert_predictions.append(pred_sentiment)
        bert_confidence.append(confidence)
    
    finbert_predictions = []
    finbert_confidence = []
    
    for sentiment in true_sentiments:
        if sentiment == 'positive':
            pred_probs = [0.85, 0.10, 0.05]
        elif sentiment == 'neutral':
            pred_probs = [0.15, 0.75, 0.10]
        else:
            pred_probs = [0.10, 0.15, 0.75]
        
        noise = np.random.normal(0, 0.03, 3)
        pred_probs = np.array(pred_probs) + noise
        pred_probs = np.clip(pred_probs, 0, 1)
        pred_probs = pred_probs / pred_probs.sum()
        
        pred_sentiment = np.random.choice(sentiments, p=pred_probs)
        confidence = pred_probs.max()
        
        finbert_predictions.append(pred_sentiment)
        finbert_confidence.append(confidence)
    
    return {
        'true_sentiments': true_sentiments,
        'bert_predictions': bert_predictions,
        'bert_confidence': bert_confidence,
        'finbert_predictions': finbert_predictions,
        'finbert_confidence': finbert_confidence
    }

def calculate_accuracy(true_labels, predictions):
    correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    return correct / len(true_labels)

def calculate_precision_recall_f1(true_labels, predictions):
    sentiments = ['positive', 'neutral', 'negative']
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for sentiment in sentiments:
        true_positives = sum(1 for t, p in zip(true_labels, predictions) if t == sentiment and p == sentiment)
        predicted_positives = sum(1 for p in predictions if p == sentiment)
        actual_positives = sum(1 for t in true_labels if t == sentiment)
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    
    return avg_precision, avg_recall, avg_f1

def calculate_metrics(true_labels, predictions):
    accuracy = calculate_accuracy(true_labels, predictions)
    precision, recall, f1 = calculate_precision_recall_f1(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def create_visualizations(data, bert_metrics, finbert_metrics):
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = ['BERT', 'FinBERT']
    accuracies = [bert_metrics['accuracy'], finbert_metrics['accuracy']]
    
    bars = axes[0, 0].bar(models, accuracies, color=['#3498db', '#2ecc71'])
    axes[0, 0].set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    bert_counts = Counter(data['bert_predictions'])
    finbert_counts = Counter(data['finbert_predictions'])
    true_counts = Counter(data['true_sentiments'])
    
    sentiments = ['positive', 'neutral', 'negative']
    bert_values = [bert_counts.get(s, 0) for s in sentiments]
    finbert_values = [finbert_counts.get(s, 0) for s in sentiments]
    true_values = [true_counts.get(s, 0) for s in sentiments]
    
    x = np.arange(len(sentiments))
    width = 0.25
    
    axes[0, 1].bar(x - width, bert_values, width, label='BERT', color='#3498db')
    axes[0, 1].bar(x, finbert_values, width, label='FinBERT', color='#2ecc71')
    axes[0, 1].bar(x + width, true_values, width, label='True', color='#e74c3c')
    
    axes[0, 1].set_title('Sentiment Classification Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xlabel('Sentiment')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(sentiments)
    axes[0, 1].legend()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bert_values = [bert_metrics['accuracy'], bert_metrics['precision'], 
                   bert_metrics['recall'], bert_metrics['f1_score']]
    finbert_values = [finbert_metrics['accuracy'], finbert_metrics['precision'], 
                      finbert_metrics['recall'], finbert_metrics['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, bert_values, width, label='BERT', color='#3498db')
    axes[1, 0].bar(x + width/2, finbert_values, width, label='FinBERT', color='#2ecc71')
    
    axes[1, 0].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xlabel('Metric')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    
    axes[1, 1].hist(data['bert_confidence'], bins=20, alpha=0.7, label='BERT', color='#3498db', density=True)
    axes[1, 1].hist(data['finbert_confidence'], bins=20, alpha=0.7, label='FinBERT', color='#2ecc71', density=True)
    axes[1, 1].set_title('Confidence Scores Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('bert_finbert_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bert_sentiment_confidence = {}
    finbert_sentiment_confidence = {}
    
    for sentiment in ['positive', 'neutral', 'negative']:
        bert_confidences = [data['bert_confidence'][i] for i, pred in enumerate(data['bert_predictions']) if pred == sentiment]
        finbert_confidences = [data['finbert_confidence'][i] for i, pred in enumerate(data['finbert_predictions']) if pred == sentiment]
        
        bert_sentiment_confidence[sentiment] = np.mean(bert_confidences) if bert_confidences else 0
        finbert_sentiment_confidence[sentiment] = np.mean(finbert_confidences) if finbert_confidences else 0
    
    x = np.arange(len(['positive', 'neutral', 'negative']))
    width = 0.35
    
    ax.bar(x - width/2, list(bert_sentiment_confidence.values()), width, label='BERT', color='#3498db')
    ax.bar(x + width/2, list(finbert_sentiment_confidence.values()), width, label='FinBERT', color='#2ecc71')
    
    ax.set_title('Average Confidence by Sentiment', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Confidence')
    ax.set_xlabel('Sentiment')
    ax.set_xticks(x)
    ax.set_xticklabels(['positive', 'neutral', 'negative'])
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('confidence_by_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data = generate_synthetic_data(1000)
    
    bert_metrics = calculate_metrics(data['true_sentiments'], data['bert_predictions'])
    finbert_metrics = calculate_metrics(data['true_sentiments'], data['finbert_predictions'])
    
    create_visualizations(data, bert_metrics, finbert_metrics)
    
    print("BERT Metrics:")
    for metric, value in bert_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nFinBERT Metrics:")
    for metric, value in finbert_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAverage Confidence Scores:")
    print(f"BERT: {np.mean(data['bert_confidence']):.4f}")
    print(f"FinBERT: {np.mean(data['finbert_confidence']):.4f}")
    
    return data, bert_metrics, finbert_metrics

if __name__ == "__main__":
    data, bert_metrics, finbert_metrics = main()