# Our Crop Prediction - Suraj K, Abhishek S, Aniket C

### Introduction About the Data :

This **dataset** is about the crop, its soil nutrient factors(Nitrogen, Phosphorous, Potassium, pH) and weather parameters(temperature, humidity, rainfall).
The goal is to predict `label` from given conditions (Classification Analysis).

There are 7 independent variables :

* `N` : ratio of Nitrogen content in soil
* `P` : ratio of Phosphorous content in soil
* `K` : ratio of Potassium content in soil
* `temperature` : temperature in degree celsius 
* `humidity` : relative humidty in percentage
* `ph` : ph value of the soil
* `rainfall` : rainfall in mm
  
Target variable :
* `label`: ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil',\n
* 'pomegranate','banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple','orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'].

Dataset Source Link :
[(https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)]

# Render Deployment Link :


# Screenshot of UI :


# Approach for the project 

1. Data Ingestion : 
    * In Data Ingestion phase the data is first read as csv. 
    * Then the data is split into training and testing and saved as csv file.

2. Data Transformation : 
    * In this phase a ColumnTransformer Pipeline is created.
    * for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * for Categorical Variables SimpleImputer is applied with most frequent strategy, then ordinal encoding performed , after this data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

3. Model Training : 
    * In this phase base model is tested . The best model found was Random Forest classifier.
    * After this hyperparameter tuning is performed on Random Forest model.
    * This model is saved as pickle file.

4. Prediction Pipeline : 
    * This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

5. Flask App creation : 
    * Flask app is created with User Interface to predict which crop is suitable for the given environment inside a Web Application.

# Exploratory Data Analysis and Model Training Approach Notebook

Link : 
