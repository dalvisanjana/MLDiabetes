# MLDiabetes

Diabetes Prediction Using SVM

This project uses machine learning techniques, specifically a Support Vector Machine (SVM) classifier, to predict whether a person has diabetes based on various medical factors.

Introduction

This project leverages the Pima Indians Diabetes Dataset to build a predictive model for diabetes. The model uses an SVM classifier to determine the likelihood of diabetes based on input features such as glucose levels, BMI, age, etc.
Dataset

The dataset used in this project is the Pima Indians Diabetes Dataset, which contains the following columns:

    Pregnancies
    Glucose
    BloodPressure
    SkinThickness
    Insulin
    BMI
    DiabetesPedigreeFunction
    Age
    Outcome

The Outcome column is the target variable, where 1 indicates diabetes and 0 indicates no diabetes.
Data Preprocessing

    Loading the Data: The dataset is loaded using Pandas.
    Data Standardization: Features are standardized using StandardScaler to ensure they have a mean of 0 and a standard deviation of 1.
    Train-Test Split: The data is split into training and testing sets using train_test_split from sklearn.model_selection.

Model Training

The SVM classifier is trained using the training data. The linear kernel is used for the SVM model.

python

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

Model Evaluation

The model's performance is evaluated using accuracy scores for both training and testing data.

python

training_data_accuracy = accuracy_score(x_train_predication, y_train)
testing_data_accuracy = accuracy_score(x_test_predication, y_test)

Prediction System

A predictive system is implemented to classify whether a person has diabetes based on their medical data. The input data is standardized before making predictions.

python

input_data = (4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
