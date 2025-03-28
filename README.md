# News_Classification_Using_NLP_and_Machine_Learning
## **Overview**

This project analyzes and classifies news articles into different categories using machine learning models. The goal is to predict the category of a news article based on its content by applying NLP techniques, feature extraction, and machine learning algorithms. Models trained include logistic regression, Naive Bayes, decision tree, random forest, and gradient boosting with performance comparisons.

---

## **Dataset Description**

- **Source:** A collection of labeled news articles from multiple domains.
- **Columns:**
  - `Article`: Text content of the news article.
  - `Category`: Category label of the article (e.g., Politics, Business, Sports, etc.).

---

## **Key Steps**

### 1. **Data Preprocessing and EDA**
- Handling missing values and duplicate entries.
- Feature engineering using tokenization, stopword removal, lemmatization, and contraction expansion.
- Text cleaning by removing punctuations, symbols, and special characters.
- Visualizing class distributions and performing train-test split.

### 2. **Feature Engineering**
- Bag of Words (BOW) and TF-IDF (unigram and bigram) vectorization for feature extraction.
- Creation of frequency dictionaries using word-category pairs.
- TSNE visualization for high-dimensional vector representation.

### 3. **Model Building and Evaluation**
- Models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, KNN, and Gradient Boosting.
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Performance comparison across vector types and classification models.
- Custom function to streamline model training, evaluation, and MLflow logging.

### 4. **MLflow Integration**
- Automated logging of model parameters, evaluation metrics, and artifacts.
- Visualization of learning curves and confusion matrices.
- Performance tracking of all models for reproducibility.

---

## **Model Performance Comparison**

| Model                        | Vector Type        | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------------------------------|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression          | Frequency Dict     | 0.953052 | -         | -      | -        | -       |
| Naive Bayes                   | BOW                | 0.974178 | 0.974684  | 0.974178 | 0.974304 | 0.998358 |
| Decision Tree                 | BOW                | 0.819249 | 0.823063  | 0.819249 | 0.819427 | 0.918991 |
| Nearest Neighbour             | BOW                | 0.7723   | 0.829554  | 0.7723  | 0.768639 | 0.928263 |
| Random Forest                 | BOW                | 0.964789 | 0.965696  | 0.964789 | 0.964860 | 0.998434 |
| Gradient Boosting             | BOW                | 0.955399 | 0.955891  | 0.955399 | 0.955487 | 0.998507 |
| Naive Bayes                   | TF-IDF Unigram     | 0.971831 | 0.972112  | 0.971831 | 0.971772 | 0.999235 |
| Decision Tree                 | TF-IDF Unigram     | 0.830986 | 0.831244  | 0.830986 | 0.830966 | 0.922558 |
| Nearest Neighbour             | TF-IDF Unigram     | 0.957746 | 0.958834  | 0.957746 | 0.957611 | 0.998362 |
| Random Forest                 | TF-IDF Unigram     | 0.943662 | 0.946625  | 0.943662 | 0.943809 | 0.998266 |
| Gradient Boosting             | TF-IDF Unigram     | 0.964789 | 0.966433  | 0.964789 | 0.964834 | 0.998197 |
| Naive Bayes                   | TF-IDF Bigram      | 0.974178 | 0.974685  | 0.974178 | 0.974256 | 0.999336 |
| Decision Tree                 | TF-IDF Bigram      | 0.821596 | 0.821733  | 0.821596 | 0.821615 | 0.921615 |
| Nearest Neighbour             | TF-IDF Bigram      | 0.964789 | 0.965075  | 0.964789 | 0.964725 | 0.998474 |
| Random Forest                 | TF-IDF Bigram      | 0.955399 | 0.957364  | 0.955399 | 0.955538 | 0.998604 |
| Gradient Boosting             | TF-IDF Bigram      | 0.962441 | 0.963359  | 0.962441 | 0.962452 | 0.998126 |

---

## **Best Model According to Each Metric**

| Metric                | Best Model                  |
|-----------------------|-----------------------------|
| Accuracy               | Naive Bayes (BOW)           |
| Precision              | Naive Bayes (TF-IDF Bigram) |
| Recall                 | Naive Bayes (BOW)           |
| F1 Score               | Naive Bayes (BOW)           |
| ROC-AUC                | Naive Bayes (TF-IDF Bigram) |

---

## **Model Insights**
- **Naive Bayes with BOW and TF-IDF Bigram** demonstrated the highest accuracy and F1 scores.
- **Random Forest and Gradient Boosting** models performed well with both BOW and TF-IDF vectors.
- **Logistic Regression** with frequency dictionaries was effective but slightly underperformed compared to other models.
- **Decision Trees and KNN** had relatively lower performance due to overfitting and sensitivity to noise.

---

## **Recommendations**
1. **Model Selection:** Deploy Naive Bayes or Random Forest with TF-IDF Bigram for best performance.
2. **Hyperparameter Tuning:** Apply Grid/Random Search for further optimization.
3. **Feature Engineering:** Explore adding N-grams, named entities, and topic modeling for enhanced feature sets.
4. **Data Augmentation:** Consider data augmentation to address class imbalance and improve model robustness.
5. **Deployment:** Use Streamlit or Flask for deploying models with REST APIs and integrate MLflow for continuous monitoring.
