
# Sentiment-Enhanced Product Recommendation System
A sentiment-enhanced recommendation system was built using fine-tuned BERT models on SST-5 for sentiment analysis and FAISS for similarity searches. It combines sentiment scores from customer reviews with product embeddings to deliver tailored recommendations.

## Overview

This project is a sentiment-enhanced recommendation system designed to improve product recommendations by combining natural language processing (NLP) techniques with embedding-based similarity search. The system analyzes customer reviews, incorporates sentiment analysis, and uses FAISS for efficient similarity search to recommend products aligned with customer preferences.

## Key Features

- **Sentiment Analysis**: 
  - Fine-tuned `distillbert-base-uncased` model on the SST-5 dataset to classify customer reviews into five sentiment classes: very negative, negative, neutral, positive, and very positive.
  
- **Embedding-Based Recommendations**:
  - Utilizes the SentenceTransformer (`all-MiniLM-L6-v2`) to compute semantic embeddings for products.
  - Incorporates sentiment scores into embeddings for sentiment-aware recommendations.
  
- **Efficient Search**:
  - Uses FAISS for fast similarity search on product embeddings to generate specific recommendations.

- **Custom Recommendation API**:
  - Provides general recommendations using GPT-based language models.
  - Refines recommendations with sentiment-enhanced embeddings.

- **Evaluation Metrics**:
  - Evaluates system performance using Precision, Recall, F1-Score, and a confusion matrix for sentiment classification.

## System Architecture

1. **Data Pipeline**:
   - Loads product metadata and customer reviews.
   - Preprocesses data, computes product embeddings, and integrates average sentiment scores into embeddings.

2. **Model Training**:
   - Fine-tunes the sentiment analysis model using the SST-5 dataset.
   - Saves models to S3 or local storage for future use.

3. **Recommendation Engine**:
   - Combines general recommendations (from a language model) with sentiment-enhanced embeddings for specific product suggestions.

4. **Evaluation**:
   - Compares performance using statistical methods such as paired t-tests to validate model improvements.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sentiment-recommender.git
   cd sentiment-recommender
   ```

2. Build the Docker image:
   ```bash
   docker build -t sentiment-recommender .
   ```

3. Run the container:
   ```bash
   docker run --gpus all -p 8080:8080 sentiment-recommender
   ```

## Usage

### 1. Recommendation API
- **Endpoint**: `/recommend`
- **Input**: JSON list of purchased products.
- **Output**: General and specific recommendations with sentiment scores.

#### Example Request:
```json
{
    "items": ["Nourishing Body Lotion", "Hydrating Face Mask Set"]
}
```

#### Example Response:
```json
{
    "general_recommendations": [
        {"title": "Gentle Exfoliating Soap", "category": "skincare", "description": "Cleanses and exfoliates."},
        {"title": "Silk-Like Leave-In Conditioner", "category": "haircare", "description": "Adds shine and smoothness."}
    ],
    "specific_recommendations": [
        {"parent_asin": "B01M9ALSWQ", "title": "Luxury Hair Oil", "score": 4.5},
        {"parent_asin": "B07DWH97YL", "title": "Organic Hair Serum", "score": 4.3}
    ]
}
```

### 2. Evaluate Model
Run the evaluation script:
```bash
python evaluate.py
```

### 3. Save and Load Models
- Save the fine-tuned sentiment model:
```python
sentiment_model.save_pretrained("/path/to/save")
```
- Load the saved model for inference:
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("/path/to/save")
```

## Data Sources

1. **Product Metadata**: Contains product titles, descriptions, and categories.
2. **Customer Reviews**: Sentiment analysis applied to customer reviews.
3. **Training Dataset**: SST-5 dataset used to fine-tune the sentiment analysis model.

## Evaluation

### Metrics:
- **Precision**: 0.4948
- **Recall**: 0.4941
- **F1-Score**: 0.4930

### Confusion Matrix:
```
[[ 58  62  17   2   0]
 [ 50 157  63  19   0]
 [  5  56  89  67  12]
 [  2   6  49 161  61]
 [  0   1  10  75  79]]
```

## Future Work

- **Advanced Fine-Tuning**: Implement LoRA or other parameter-efficient fine-tuning methods.
- **User Behavior Data**: Incorporate user behavior features such as click rates, time spent, and purchase history.
- **Scalability**: Extend support for larger datasets and multi-language sentiment analysis.
- **Explainable AI**: Add interpretable explanations for recommendations.

## Dependencies

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- Flask
- Docker
- AWS S3 (optional)

## Contributors

- Tirus Wagacha
- Rafat Shahrair

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

--- 

You can further customize this file with additional details as needed.
