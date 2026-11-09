# Predicting term-deposit subscriptions: Logistic regression and random forest

## Question
Which bank clients are most likely to subscribe to a term deposit after a marketing call, so that a campaign can call the right people first?

## Data
Bank Marketing dataset (Moro, Cortez & Rita, 2014), UCI Machine Learning Repository: phone marketing campaigns of a Portuguese bank (`bank-full.csv`, about 45,000 clients, 16 variables). About 11.7% of clients subscribed, so the classes are imbalanced.

## Method
1. **Data checks and cleaning:** hidden missing values (`unknown`) kept as a category, `pdays = -1` handled with a clear "previously contacted" indicator.
2. **Data leakage:** the variable `duration` (call length) is only known after the call, so it was removed. *(Note: keeping it would raise the ROC-AUC artificially to ~0.95, which would be completely misleading for a predictive tool).*
3. **Train / test split:** 80 / 20, stratified.
4. **Pipelines & Preprocessing:** Use of Scikit-Learn `Pipeline` and `GridSearchCV` to handle categorical encodings, scaling, and hyperparameter tuning cleanly without data leakage.
5. **Models compared with cross-validation:** baseline, Logistic Regression, and Random Forest. Metrics: ROC-AUC, Average Precision, Recall, Precision (accuracy is intentionally avoided because of class imbalance).
6. **Model Interpretation:** Analyzed via Logistic Regression coefficients (log-odds and odds ratios) and Random Forest Permutation Importance (drop in ROC-AUC).
7. **Final evaluation** on the untouched test set, followed by threshold, lift, and targeting analysis.

## Results

| Model | ROC-AUC (CV) | Recall | Precision |
| :--- | :--- | :--- | :--- |
| **Baseline** | 0.50 | 0.00 | 0.00 |
| **Logistic Regression** | 0.78 | 0.35 | 0.52 |
| **Random Forest** | 0.80 | 0.39 | 0.55 |

On the test set, the **Logistic Regression** reached a ROC-AUC of **0.78** (against **0.80** for the Random Forest). Calling only the top 20% of clients ranked by the model would reach about **65%** of the subscribers (a lift of **3.25**).

## Recommendation
* **Model Choice:** I recommend **Logistic Regression** because it offers almost the same predictive performance as the Random Forest while remaining fully transparent, explainable, and easily auditable by risk and compliance teams.
* **Main drivers of subscription:** 
  * **Timing (`month`):** Avoid dead periods like January; focus efforts on high-performing months such as March, September, and October.
  * **Historical engagement (`poutcome_success`):** Clients who accepted a previous campaign are over 4 times more likely to say yes again (odds ratio of 4.25).
* **Threshold & Targeting:** At a default operating threshold of 0.35, the model successfully calls 25% of clients to capture 72% of total subscribers. The final threshold should be adjusted by the marketing team based on the exact cost of a phone call versus the lifetime value of a subscription.

## Limits
- **One bank and one period:** Portugal, 2008-2010; results may not transfer directly to another market or economic period.
- **No cost model:** The decision threshold is discussed conceptually, not optimised with real-world campaign costs and revenues.
- **Simple data preparation:** `balance` is skewed and could benefit from a log-transformation; temporal variables like `day` and `month` are treated as categorical rather than cyclical.
- **No probability calibration:** The scores rank clients effectively for targeting, but they are raw model outputs rather than exact mathematical probabilities.
- **Next steps:** Test gradient boosting architectures (e.g., LightGBM), apply probability calibration, and monitor model drift over time with incoming campaign data.

## How to run
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook bank_marketing_analysis.ipynb
