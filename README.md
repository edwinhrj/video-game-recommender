# Video Game Recommendation System

## Project Overview

Users often face decision fatigue due to an overwhelming number of choices on gaming platforms like Steam. This can reduce engagement and retention. While Steam’s Interactive Recommender uses implicit signals such as playtime, it lacks explainability and does not fully utilize textual data like reviews and game descriptions.

Our goal is to enhance recommendation relevance and explainability by integrating textual implicit signals and explicit features such as user reviews and generated ratings. We experiment with six models spanning content-based and collaborative filtering approaches:

---

- Stacked Logistic Regression (SLR)
- Sentence-BERT (SBERT)
- Wide & Deep (W&D)
- Deep Cooperative Neural Network (DeepCoNN)
- Light Graph Convolutional Network (LightGCN)
- Neural Collaborative Filtering (NCF) with enhancements

---

## Feature Engineering

### Predicted Rating Formation

Since Steam lacks an explicit rating system, we used a HuggingFace transformer model fine-tuned on Amazon review sentiment data to convert user reviews into predicted ratings (`pred_rating`). This enabled us to generate a proxy for explicit user feedback.

### LDA Topic Modeling

Applied Latent Dirichlet Allocation (LDA) on game descriptions to extract latent topics that capture overlapping game characteristics. Using 30 topics optimized by coherence scores, these topic distributions enriched item-specific features and improved contextual understanding.

### K-Means Clustering

Users and games were clustered into groups based on numerical features:

- Users: grouped into 6 clusters using features like number of games owned, total playtime, and review count.
- Games: grouped into 5 clusters using price, DLC count, required age, and review counts.

These clusters were used as categorical inputs to supplement the Wide & Deep model.

---

## Models and Performance

| Model                    | Precision@4 | Recall@4 | F1@4  | NDCG@4 |
|--------------------------|-------------|----------|-------|--------|
| SLR (Numeric Features)   | 4.94%       | 4.00%    | 0.043 | 0.049  |
| SLR (TF-IDF Features)    | 2.83%       | 2.17%    | 0.024 | 0.042  |
| SLR (Stacked Features)   | 2.82%       | 2.18%    | 0.024 | 0.034  |
| SBERT                    | 1.06%       | 0.97%    | 0.010 | 0.012  |
| Wide & Deep              | 2.11%       | 1.67%    | 0.018 | 0.020  |
| DeepCoNN                 | 3.00%       | 3.00%    | 0.030 | 0.018  |
| LightGCN                 | **12.3%**   | **22.4%**| 0.158 | 0.198  |
| NCF                      | 3.86%       | 2.79%    | 0.032 | 0.039  |
| Enriched NCF             | 4.50%       | 3.26%    | 0.037 | 0.047  |

### Content-Based Models

- **Stacked Logistic Regression (SLR):** Combines numerical features with TF-IDF vectors of game descriptions. The stacked approach balances sparse textual data with numerical features.
- **SBERT:** Uses sentence-level embeddings of game descriptions and reviews for semantic similarity matching, leveraging FAISS and HNSW libraries for efficient nearest neighbor search.
- **Wide & Deep:** Integrates memorization of explicit features (wide) with learned embeddings of user interaction history (deep) for personalized recommendations.

### Collaborative Filtering Models

- **DeepCoNN:** Learns user and item representations from textual data (item descriptions and user reviews) via convolutional neural networks.
- **LightGCN:** Leverages graph convolutional networks to model the social network aspect of Steam users and games, capturing community structures.
- **Neural Collaborative Filtering (NCF):** Combines matrix factorization with deep neural layers to model complex user-item interactions. The enriched variant adds item metadata for improved performance.

---

## Key Contributions

- **Creative Modelling:** Developed custom content-based and enriched collaborative filtering models tailored for the video game domain.
- **Use of Explicit Signals:** Incorporated predicted ratings from reviews and textual data to increase explainability.
- **Advanced Feature Engineering:** Employed LDA topic modeling and clustering to capture latent item and user characteristics.
- **Similarity Search Optimization:** Implemented FAISS for efficient approximate nearest neighbor search on high-dimensional embeddings.
- **Addressing Data Sparsity:** Established user/item interaction thresholds to improve data density from 0.61% to 15.4%, enabling more effective model training.

---

## Evaluation and Results

- Used a temporal train-test split simulating real-world conditions.
- The predicted ratings serve as ground truth but may introduce some noise due to domain mismatch of the transformer model.
- LightGCN outperformed other models, likely because it captures community relationships in the user-game interaction graph.
- Popularity bias exists due to thresholding, favoring popular games but necessary for denser graph structures.
- Sparse TF-IDF features showed weaker performance in logistic regression models, potentially due to high dimensionality and feature correlation.
- The Wide & Deep model's limitations stemmed from missing key user demographic and impression data, restricting full generalizability.

---

## Datasets

- [Top 1000 Steam Games](https://www.kaggle.com/datasets/joebeachcapital/top-1000-steam-games)
- [2021 Steam Reviews](https://www.kaggle.com/datasets/najzeko/steam-reviews-2021)

---

## References

- Cheng, H. T. et al., "Wide & Deep Learning for Recommender Systems," 2016. [arXiv](https://arxiv.org/abs/1606.07792)
- Zheng, L., Noroozi, V., & Yu, P. S., "DeepCoNN," 2017. [arXiv](https://doi.org/10.48550/arXiv.1701.04783)
- Schwartz, B., "More Isn’t Always Better," Harvard Business Review, 2006. [HBR](https://hbr.org/2006/06/more-isnt-always-better)
- Jegou, H., Douze, M., & Johnson, J., "FAISS: A Library for Efficient Similarity Search," Meta Engineering, 2018. [Article](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)

---


