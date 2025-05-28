
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
| Model Type               | Input Features         | Classes       | Accuracy | F1 Score |
|--------------------------|------------------------|---------------|----------|----------|
| Multinomial Naive Bayes  | Vectorizer + Metadata  | Discrete      | 26.9%    | 27%      |
| Bayesian Neural Net      | BERT + Metadata        | 6-way         | 1.6%     | 16.9%    |
| Bayesian Neural Net      | BERT + Metadata        | Binary        | 54.3%    | 55.1%    |
| Deterministic Model      | BERT + Metadata        | 6-way         | 27.3%    | 26.7%    |
| Deterministic Model      | BERT + Metadata        | Binary        | 69.4%    | 54.5%    |
| BNN + Pretrained Guide   | BERT + Metadata        | Binary        | 69%      | 66.7%    |

### Architecture Details

Multinomial Naive Bayes:
- Text Encoder: Term Frequency Inverse Document Frequency (TFIDF) vectorizer
- Metadata Encoding: Label encoding
- MNB Implementation: Sklearn (TfidfVectorizer + MultinomialNB)

Bayesian Neural Net:
- Text Encoder: Pre-trained [`bert-base-uncased`](https://arxiv.org/abs/1810.04805)
- Metadata Encoding: Label encoding + MinMax scaling
- Multimodal Fusion: Concatenation of BERT embedding and encoded metadata
- BNN Implementation:  
  - Pyro + PyTorch  
  - Priors: `Normal(0, 0.5)` on all weights and biases  
  - Inference: Stochastic Variational Inference (SVI)

### BNN Model Finetuning

- Priors: `Normal(0, 0.5)` -> `Normal(0., 0.05)`
- DataLoader: `batch_size=8` -> `batch_size=12`
- Hidden Layers: `128` -> `256`
- ClippedAdam Optimizer: learning rate = `1e-3` -> `5e-4`

*Model Improvement: Accuracy +4.28%, F1 Score +3.6%*

### Pre-training Model Guide using Deterministic Model

Guide: `AutoDiagonalNormal`
- Represents each parameter's posterior as an independent normal distribution
- Initially, this guide randomizes the initial posterior mean approximations, which may lead to approximations far from the true posterior.
- Using a deterministic model to pre-train the weights and biases will give the Guide a better approximation of the posterior, which can improve model performance.

*Model Improvement (compared to Finetuned BNN): Accuracy +14.7%, F1 Score +11.6%*

---

## Key Observations

- MNB model improves with increase text columns but not significantly.
- Final MNB model combines training and validation set performs best suggesting increasing metadata improves model. 
- Binary classification significantly outperforms 6-way classification for both BNN and deterministic models.
- BNN underperforms in the 6-class setting, producing nearly uniform class distributions and high predictive entropy.
- BNN shows high uncertainty even in validation.

---

## Uncertainty Estimation (BNN)

Confidence: 

| Prediction Confidence Distribution                                                        | Prediction Confidence Distribution (with Pre-trained Guide)                               |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| ![image](https://github.com/user-attachments/assets/fe5a262a-230d-4465-99cb-35bdeb6f0b2e) | ![image](https://github.com/user-attachments/assets/f15ef9e0-dd73-4412-9eb8-a4b968d5e226) |

- There is a more pronounced long-tail towards high confidence, demonstrating that a pre-trained guide makes the model more confident in its prediction overall.

Entropy: High across most samples.

| Predictive Entropy Distribution                                                           | Predictive Entropy Distribution (with Pre-trained Guide)                                  |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| ![image](https://github.com/user-attachments/assets/581a8602-933e-48d1-a37d-05ec62a5dd8a) | ![image](https://github.com/user-attachments/assets/9571e447-eab2-434e-97b3-2fe452462382) | 
  
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
