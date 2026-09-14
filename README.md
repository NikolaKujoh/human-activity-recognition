# Human Activity Recognition using Classical ML

Replicating and extending the results from Anguita et al. (2013) on the UCI HAR dataset, benchmarking classical machine learning models for human activity classification from smartphone sensor data.

## Project Description

Six everyday activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) are classified using 561 time- and frequency-domain features extracted from accelerometer and gyroscope signals of a waist-mounted smartphone. Logistic Regression, SVM, Random Forest, and Gradient Boosting are trained and evaluated, with results benchmarked against the original Anguita et al. (2013) paper.

Full write-up (EDA, methodology, results, and error analysis) is available in [`Izveštaj.pdf`](./Izveštaj.pdf).

## Dataset

UCI Human Activity Recognition Using Smartphones Dataset  
Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

10,299 samples from 30 subjects, pre-split into train (7,352 samples, 21 subjects) and test (2,947 samples, 9 subjects) sets, split at the subject level to prevent data leakage.

After downloading, place the data in a `data/` folder in the project root.

## Models & Results

| Model | Test Accuracy | Reference |
|---|---|---|
| Logistic Regression | 95.45% | — |
| SVM (RBF) | 95.42% | Anguita et al.: ~96% |
| Gradient Boosting | 94.03% | — |
| Random Forest | 92.98% | — |
| Majority class (baseline) | 18.22% | — |

The main source of error across all models is confusion between SITTING and STANDING — a known limitation also reported by Anguita et al. See the report for full confusion matrices and analysis.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb   # EDA
jupyter notebook models.ipynb     # model training & evaluation
```

## License

Code in this repository is released under the [MIT License](./LICENSE). The UCI HAR dataset is licensed separately under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by its original authors.

## References

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J.L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. *ESANN 2013*.