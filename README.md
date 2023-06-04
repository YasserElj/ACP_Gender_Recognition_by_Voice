# PCA Analysis on Voice Gender Dataset

Perform PCA (Principal Component Analysis) on the Voice Gender dataset to classify voices as male or female based on acoustic properties.

## [Dataset](https://www.kaggle.com/datasets/primaryobjects/voicegender)


The dataset consists of 3,168 recorded voice samples collected from male and female speakers. Preprocessing techniques using R packages seewave and tuneR were applied, analyzing the frequency range of 0Hz-280Hz (human vocal range).

## PCA Analysis

Apply PCA to extract essential information from the dataset. Focus on the top 5 variables that capture approximately 80% of the information required for classification.

## Model Comparison


Compare model performance on the entire dataset versus using only the 5 selected variables. Evaluate accuracy and F1-scores using the XGBoost classifier.

## Results

The model trained on the entire dataset slightly outperforms the model using only 5 variables. However, the difference in performance is minimal, indicating that the selected variables contain significant classification information.

### The entire dataset

```python
x_train = train.iloc[:, :-1]
y_train = train["label"]
x_test = test.iloc[:, :-1]
y_test = test["label"]

model = xgboost.XGBClassifier()
classify(model,x_train,y_train,x_test,y_test)
```
```

              precision    recall  f1-score   support

      female     0.9730    0.9832    0.9781       476
        male     0.9830    0.9726    0.9778       475

    accuracy                         0.9779       951
   macro avg     0.9780    0.9779    0.9779       951
weighted avg     0.9780    0.9779    0.9779       951
```
### The 5 selected variables
```python
x_train2 = train[["meanfun","sd","median","Q25","Q75"]]
y_train2 = train["label"]
x_test2 = test[["meanfun","sd","median","Q25","Q75"]]
y_test2 = test["label"]

model = xgboost.XGBClassifier()
classify(model,x_train2,y_train2,x_test2,y_test2)
```
```

              precision    recall  f1-score   support

      female     0.9746    0.9664    0.9705       476
        male     0.9666    0.9747    0.9706       475

    accuracy                         0.9706       951
   macro avg     0.9706    0.9706    0.9706       951
weighted avg     0.9706    0.9706    0.9706       951
```
## Conclusion

Using just the 5 variables can simplify the recognition model without sacrificing much in terms of performance.

#



## Dependencies

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Instructions

1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter Notebook to execute the PCA analysis and model comparison.


