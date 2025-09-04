# NLP: Text Classification Analysis

Comparison of traditional ML and deep learning approaches for 5-class text document classification.

## Quick Results

| Model | Accuracy | F1-Score | Speed |
|-------|----------|----------|-------|
| **Naive Bayes** | **98.2%** | **98.2%** | Fast |
| Linear SVM | 97.8% | 97.7% | Fast |
| DistilBERT | 97.3% | 97.3% | Slow |
| LSTM | 90.6% | 90.5% | Medium |

## Key Finding
Traditional ML (Naive Bayes) outperformed deep learning models with 98.2% accuracy while being 100x faster to train.

## Installation & Usage

```bash
pip install scikit-learn tensorflow transformers nltk
python text_classification.py
```

## Models Tested

### Traditional ML
- **Naive Bayes**: Probabilistic word frequency approach
- **Linear SVM**: Maximum margin classification with TF-IDF

### Deep Learning  
- **LSTM**: Recurrent network for sequential text processing
- **DistilBERT**: Pre-trained transformer model

## Implementation Details

### Data Processing
- **Dataset**: 2,225 documents, 5 classes
- **Preprocessing**: Text cleaning, tokenization, lemmatization
- **Features**: TF-IDF vectorization (traditional ML), embeddings (deep learning)

### Key Insights
- **TF-IDF effectiveness**: Traditional features highly effective for this dataset
- **Perfect ROC curves**: Naive Bayes and SVM achieved AUC = 1.00
- **Training efficiency**: Traditional models trained in seconds vs. hours
- **Dataset characteristics**: Well-separated classes favor simpler approaches

## Technical Highlights
- Complete preprocessing pipeline with NLTK
- ROC curve analysis for all models
- Systematic performance comparison across ML paradigms
- Production-ready model evaluation framework


---

**Conclusion**: For this well-structured text classification task, traditional ML provides optimal balance of performance, interpretability, and computational efficiency.
