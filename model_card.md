# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was built by Ryan Schuler for the WGU D501 Machine Learning DevOps course (Udacity project "Deploying a Machine Learning Model with FastAPI"). It is a scikit-learn `RandomForestClassifier` (version 1.5.1) with 100 trees, a maximum depth of 20 and `random_state=42`. The eight categorical features are one-hot encoded with a `OneHotEncoder`, the label is encoded with a `LabelBinarizer`, and the six numeric features are used as they are. The trained model and encoder are saved as `model/model.pkl` and `model/encoder.pkl`, and the model is served through a FastAPI app in `main.py`.

## Intended Use

The model predicts whether a person's yearly income is above or below $50K based on census attributes like age, education, occupation and hours worked per week. It was built to practice the full workflow of training, testing, monitoring and deploying a model. It is meant for learning and demonstration only. It should not be used to make real decisions about people, such as hiring, lending, housing or pay.

## Training Data

The data is the UCI Census Income ("Adult") dataset, extracted from the 1994 US Census database (https://archive.ics.uci.edu/dataset/20/census+income). The provided `census.csv` has 32,561 rows and 15 columns: 14 features and the `salary` label. About 76% of rows are `<=50K` and 24% are `>50K`, so the classes are unbalanced. Missing values show up as `?` in `workclass`, `occupation` and `native-country`, and they were kept as their own category.

The data was split 80/20 into training and test sets with `train_test_split`, stratified on `salary` with `random_state=42`. The training set has 26,048 rows.

## Evaluation Data

The evaluation data is the 20% test split, 6,513 rows, which the model never saw during training. It was processed with the encoder and label binarizer fitted on the training data.

## Metrics

The model was evaluated with precision, recall and F1 score, where `>50K` is the positive class. On the test set the model scored:

- Precision: 0.7857
- Recall: 0.6103
- F1: 0.6870

Precision is fairly high, so when the model predicts `>50K` it is usually right. Recall is lower, which means it misses about 39% of the people who actually earn over $50K.

Performance was also measured on every value of every categorical feature, and the full results are in `slice_output.txt`. A few examples: F1 is 0.6793 for women and 0.6884 for men, and by race it ranges from 0.5333 (Amer-Indian-Eskimo, 73 rows) to 0.6914 (White, 5,533 rows). Small slices, like `workclass: Never-worked` with a single row, give scores that aren't meaningful.

## Ethical Considerations

The dataset includes sensitive attributes such as race, sex and native country, and the model uses them as inputs. Because the data reflects income gaps that existed in 1994, the model can repeat those patterns in its predictions. The slice results show performance isn't even across groups, and recall is lower for some smaller groups, so the model is more likely to miss high earners in those groups. Anyone using this model should look at the slice metrics before trusting a prediction for a specific group.

## Caveats and Recommendations

The data is more than 30 years old, so income levels, job categories and the $50K cutoff don't reflect today's economy. Some groups have very few rows, which makes their metrics unreliable. The hyperparameters were set by hand and not tuned. To improve the model, I'd recommend tuning the hyperparameters with cross-validation, adjusting the decision threshold or class weights to raise recall, retraining on more recent data, and checking fairness metrics across groups before any real use.
