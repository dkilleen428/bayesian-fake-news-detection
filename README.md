
# Fake News Classification with Bayesian Neural Networks  
**Team: Regularize This! – Danielle Killeen, Paul Pham, Alvin Yao**  
**Course: Bayesian Machine Learning with Generative AI Applications**

## Overview
In the age of rapid digital misinformation, particularly in politics, detecting fake news is a critical task. This project explores a Bayesian Neural Network (BNN)-based pipeline to classify political statements by truthfulness using the LIAR dataset. We utilize textual content via BERT embeddings and speaker-related metadata to build a multimodal classifier capable of not only predicting truthfulness but also quantifying uncertainty. We also employed Multinomial Naive Bayes, a variation of Naive Bayes classification algorithm to establish a baseline. MNB is specially designed for discrete data particularly text data and is commonly used in NLP for document classification.

---

## Dataset
**LIAR dataset** from [PolitiFact](https://arxiv.org/abs/1705.00648v1):  
- 12,836 manually labeled political statements  
- 6 fine-grained classes:  
  `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`  
- Includes metadata: speaker, job, state, party, context, and credibility history

---

## Methodology

### Modeling Approaches
| Model Type                | Input Features         | Classes       | Accuracy | F1 Score |
|--------------------------|------------------------|---------------|----------|----------|
| Multinomial Naive Bayes  | Vectorizer + Metadata  | Discrete      | 26.9%   | 27%       |
| Bayesian Neural Net      | BERT + Metadata        | 6-way         | 1.6%    | 16.9%    |
| Bayesian Neural Net      | BERT + Metadata        | Binary        | 50.0%    | 51.5%    |
| Deterministic Model      | BERT + Metadata        | 6-way         | 27.3%    | 26.7%    |
| Deterministic Model      | BERT + Metadata        | Binary        | 69.4%    | 54.5%    |

### Architecture Details
---
- Text Encoder: Term Frequency Inverse Document Frequency (TFIDF) vectorizer
- Metadata Encoding: Label encoding
- MNB Implementation: Sklearn (TfidfVectorizer + MultinomialNB)
---
- Text Encoder: Pre-trained [`bert-base-uncased`](https://arxiv.org/abs/1810.04805)
- Metadata Encoding: Label encoding + MinMax scaling
- Multimodal Fusion: Concatenation of BERT embedding and encoded metadata
- BNN Implementation:  
  - Pyro + PyTorch  
  - Priors: `Normal(0, 0.5)` on all weights and biases  
  - Inference: Stochastic Variational Inference (SVI)

---

## Key Observations

- MNB model improves with increase text columns but not significantly.
- Final MNB model combines training and validation set performs best suggesting increasing metadata improves model. 
- Binary classification significantly outperforms 6-way classification for both BNN and deterministic models.
- BNN underperforms in the 6-class setting, producing nearly uniform class distributions and high predictive entropy.
- Deterministic model with frozen BERT achieved the best performance in both binary and multiclass settings.
- BNN shows high uncertainty even in validation, suggesting poor class separation or over-regularization due to wide priors.

---

## Uncertainty Estimation (BNN)

- Confidence: Most predictions were close to 1/6 probability, indicating the model wasn’t confident.
- Entropy & Variance: High across most samples.
- Interpretation: Model learned to “be uncertain”—valuable for abstaining from ambiguous cases but detrimental to accuracy without calibration.

---

## Tools & Libraries
- Pyro, Torch, transformers, scikit-learn, matplotlib
- Python 3.11, PyTorch 2.6
- BERT tokenizer and embeddings from Hugging Face

---

## Limitations & Future Work

### Modeling Improvements
- Unfreeze BERT layers for domain adaptation
- Explore Monte Carlo Dropout as a lighter-weight Bayesian approach
- Try tighter priors (e.g., `Normal(0, 0.1)`)
- Investigate hierarchical BNNs or Bayesian ensembles
- MNB is more efficient with very high text datasets with thousands of words.


### Data & Evaluation
- Augment data with additional political datasets or synthetic samples
- Use class weights or focal loss to handle label imbalance
- Add confusion matrices, per-class F1 scores, and calibration curves for deeper insight

---

## Conclusion
This project demonstrates that while Bayesian approaches offer rich uncertainty modeling, they may require more data, tighter priors, or better optimization to compete with simpler deterministic methods in terms of raw accuracy. Still, BNNs provide valuable insight into model confidence—a critical aspect in high-stakes domains like misinformation detection. 

---

## Resources
- LIAR Dataset: https://arxiv.org/abs/1705.00648v1  
- BERT Paper: https://arxiv.org/abs/1810.04805  
- Pyro Framework: https://pyro.ai/
- https://mashimo.wordpress.com/2019/07/13/classification-metrics-and-naive-bayes/
