# Diabetes-Detection
Experiment with Forward neural network and macine learning models to detect possibility of diabetes in patients.

### Dataset
* [Data](https://github.com/Shreets/Diabetes-Detection/blob/main/data/diabetes.csv)

The data consists of 769 sapmles from patiesnts tested for diabetes. The other features present in the data are 
* Pregnancies: Number of times pregnant
* Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
* BloodPressure: Diastolic blood pressure (mm Hg)
* SkinThickness: Triceps skin fold thickness (mm)
* Insulin: 2-Hour serum insulin (mu U/ml)
* BMI: Body mass index (weight in kg/(height in m)^2)
* DiabetesPedigreeFunction: Diabetes pedigree function
* Age: Age (years)
The data is labeled and has the class information in column `Outcome` where non-diabetic and diabetic patients are denoted by laebls 0 and 1 respectively.

## Data processing
The data provided is unbalanced. The number of samples for diabetic patient is half the number of non-diabetic ones. This may cause bias in the model
```
Number of patients with Diabetes : 268
Number of patients without Diabetes : 500
```
To solve this we have used opersampling method i.e *selecting random points from the minority class and duplicating them to increase the number of data points in the minority class* to create a balance in data before we use it to train the model.



### Models
* [Diabetes_Prediction_machine_learning.ipynb](https://github.com/Shreets/Diabetes-Detection/blob/main/model/Diabetes_Prediction_machine_learning.ipynb)
* [Diabetes_Prediction_using_feed_forward_neural_network.ipynb](https://github.com/Shreets/Diabetes-Detection/blob/main/model/Diabetes_Prediction_using_feed_forward_neural_network.ipynb)

For feed forward network, we have used following components

Dense Layer : 16 neurons

Relu activation function : `if x <= 0 --> 0, x > 0 --> x`

Sigmoid activation function: `if x <= 0.5 --> 0, x > 0.5 --> x`

where sigmoid activation function has been used for binary classification ; it maps the input to probability of weather or not something belongs to a single class. (maps to 0 or 1)

For machine learning we have used three different models and compared their results to select the one best performance
1. Logistic Regression
2. Decision Tree
3. Random Forest 

The best performance was achieved using the ensemble method with Random forest classifier. 

Neural networks require a large volume of data to derive the right patterns for the model to learn from, hence the FFN doesnt have the best of performance but it compares to the result of ML models like Logistic regression and decision tree. Deep learning techniques with deeper layes were not experimented with since the available dataset is not enough.


