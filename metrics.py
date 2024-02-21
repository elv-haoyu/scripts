import numpy as np


def calculate_dcg(ratings):
    return sum((2**rating - 1) / np.log2(index + 2) for index, rating in enumerate(ratings))

def calculate_ndcg(group):
    ratings = group['avg_rating'].tolist()
    sorted_ratings = sorted(ratings, reverse=True)
    
    dcg = calculate_dcg(ratings)
    idcg = calculate_dcg(sorted_ratings)
    
    return dcg / idcg if idcg > 0 else 0