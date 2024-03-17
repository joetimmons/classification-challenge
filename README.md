# classification-challenge

## Spam Detector
This project demonstrates how to build a spam detector using machine learning techniques. It compares the performance of a Logistic Regression model and a Random Forest Classifier model on a spam dataset.

## Dataset
The dataset used in this project is the Spambase dataset from the UCI Machine Learning Library. It contains various features extracted from email messages, along with a label indicating whether the email is spam or not.

The dataset is located at: https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv

## Requirements
Python 3.x
pandas
scikit-learn
joblib


## Notebook Structure
The Jupyter Notebook spam_detector.ipynb contains the following sections:

Retrieve the Data: Imports the spam dataset using pandas and displays the resulting DataFrame.
Predict Model Performance: Provides a prediction and justification for which model (Logistic Regression or Random Forest Classifier) is expected to perform better.
Split the Data into Training and Testing Sets: Splits the data into training and testing sets using train_test_split from scikit-learn.
Scale the Features: Scales the feature data using StandardScaler from scikit-learn.
Create and Fit a Logistic Regression Model: Creates a Logistic Regression model, fits it to the training data, makes predictions on the testing data, and evaluates the model's accuracy.
Create and Fit a Random Forest Classifier Model: Creates a Random Forest Classifier model, fits it to the training data, makes predictions on the testing data, and evaluates the model's accuracy.
Evaluate the Models: Compares the performance of the Logistic Regression and Random Forest Classifier models and provides an analysis of the results.

## Results
The accuracy scores of the Logistic Regression and Random Forest Classifier models are printed in the notebook. The model that performs better is identified, and the results are compared to the initial prediction.

The trained Logistic Regression model is saved as logreg_model.pkl, and the predictions are saved in your_predictions_path.csv.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.