# Human Activity Recognition using Classical ML

Replicating and extending the results from Anguita et al. (2013) using classical machine learning methods on the UCI HAR dataset.

## Project Description

We benchmark Logistic Regression, SVM, and ensemble methods on smartphone sensor data to classify human physical activities, comparing our results against published literature.

## Dataset

UCI Human Activity Recognition Using Smartphones Dataset  
Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

After downloading, place the data in a `data/` folder in the project root.

## Models

- Logistic Regression (baseline)
- SVM with RBF kernel
- Random Forest
- Gradient Boosting
- TBD

*Reference (Anguita et al. 2013): ~96% SVM accuracy*

## How to Run

```bash
pip install -r requirements.txt
python learning.py
```

## References

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J.L. (2013).
A Public Domain Dataset for Human Activity Recognition Using Smartphones.
ESANN 2013.