from typing import Any, Dict, List

import torch
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load tokeniser, model, and use pipeline for inference
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
sentiment_analyser = pipeline(
    task="sentiment-analysis", 
    model=model,
    tokenizer=tokenizer, 
    device=-1 
) # type: ignore

# Define a function to apply sentiment analysis
def analyse_company_sentiments(obj: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Apply sentiment analysis to a list of review objects and enrich each object with 'score' and 'label'.

    :param obj: A list of dictionaries, each expected to have a 'content' key.
    :param batch_size: The number of reviews to process in each batch.
    :return: A list of dictionaries with added 'score' and 'label' from sentiment analysis.
    """
    # Extract content for sentiment-analysis from object
    review_contents = [review["content"] for review in obj]
    
    all_results = []
    # torch.set_num_threads() should be set once, not in a loop
    torch.set_num_threads(2)

    # Process the texts in batches
    for i in trange(0, len(review_contents), batch_size):
        batch_text = review_contents[i:i + batch_size]
        batch_obj = obj[i:i+batch_size]
        with torch.no_grad():
            analysis_results = sentiment_analyser(batch_text)
            
            # Enrich object with sentiment score and label
            batch_obj_with_sentiment = [
                {**review, "score": result["score"], "label": result["label"]}
                for review, result in zip(batch_obj, analysis_results)
            ]
            all_results.extend(batch_obj_with_sentiment)

    return all_results


def create_positive_negative_reviews_dict(reviews: List[Dict[str, Any]], company: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Aggregates positive and negative reviews for a single company group.

    :param reviews: List of review dictionaries with 'label', 'company', and 'content' keys.
    :param company: The company name to filter reviews for.
    :return: A dictionary with the company name as key, containing 'POSITIVE' and 'NEGATIVE' lists.
    """
    pos_content: List[str] = []
    neg_content: List[str] = []

    for review in reviews:
        if review.get("company") != company:
            continue  # Skip reviews from other companies

        content = review.get("content", "")
        label = review.get("label")

        if label == "POSITIVE":
            pos_content.append(content)
        elif label == "NEGATIVE":
            neg_content.append(content)

    return {
        company: {
            "POSITIVE": pos_content,
            "NEGATIVE": neg_content
        }
    }


# Create a function to concatenate Positive and negative reviews
def concatenate_reviews(obj: List[Dict[str, Dict[str, List[str]]]], company: str) -> Dict[str, Dict[str, str]]: 
    results = {}
    for review in obj: 
        company_review = review.get(company)
        if company_review is not None: 
            concatenated_pos_reviews: str = "\n".join([f"{i}. {rev}" for i, rev in enumerate(company_review.get("POSITIVE", []), start=1)])
            concatenated_neg_reviews: str = "\n".join([f"{i}. {rev}" for i, rev in enumerate(company_review.get("NEGATIVE", []), start=1)])
            results[company] = {"POSITIVE": concatenated_pos_reviews, "NEGATIVE": concatenated_neg_reviews}
    
    return results