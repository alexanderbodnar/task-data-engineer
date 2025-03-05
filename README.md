# DTSE Data Engineer (ETL) assignment

You are given this repository from the data science team. It contains a Python script (`main.py`) that generates a model, stores it in a file (`model.joblib`) and is later used to generate a house price prediction based on the house property parameters.

Your task is to provide an automated solution that processes input data, transforms the data to required format of a model and stores the results of transformation as well as model predictions in database.

## Mandatory Task
1. Create a functionality in python (understand as script or multiple scripts, classes, functions, etc.) that processes input (`housing.csv`), transforms it, saves it to db, provides it to a model and saves the predictions in db again. Usually data is provided to us in nonstandardized format by customer and we need to be sure it's processed correctly for our model to be able to generate predictions.
2. Demonstrate with a provided test data that it works and how it should be used (`housing.csv`)
3. After installing new libraries, update requirements.txt file
4. Create a new readme file describing new functionality and how to use the processing pipeline for preparation of input for data modeling

## Optional Tasks and topics for discussion
1. Logging - how and why would you implement it?
2. Tests - how and why would you implement it?
3. Exception handling - how and why would you implement it?
4. API - how and why would you implement it?

## Submitting your solution
The preferred form of submission is to place the whole solution in a public GitHub repository and send us a link. Both the dataset and model are distributed under the public license. If you don't wish to display your solution publicly, you can add repository view permissions to the email kosztolanyistefan@gmail.com or send a zip archive with the code to the same email address.

## Notes
* You should not generate any new model. Use the model provided in the `model.joblib` file.
* If you use a database, is should be part of your solution as a file.
* If something is unclear or you run into any technical diffilcuties, feel free to contact us.
* Python 3.9.13 was tested with the solution, thus this version is recommended to use. Use a different version at your own risk.

### Files
* `main.py` - sample script that generates and uses the model for predictions
* `model.joblib` - the computed model you should use 
* `housing.csv` - data file to process and apply a model to it for creating predictions
* `requirements.txt` - pip dependencies

## Sample outputs
You can validate you predictions on these sample inputs and expected outputs.

Input 1:
```
longitude: -122.64
latitude: 38.01
housing_median_age: 36.0
total_rooms: 1336.0
total_bedrooms: 258.0
population: 678.0
households: 249.0
median_income: 5.5789
ocean_proximity: 'NEAR OCEAN'
```

Output 1: `320201.58554044`

-----------------------------------

Input 2:
```
longitude: -115.73
latitude: 33.35
housing_median_age: 23.0
total_rooms: 1586.0
total_bedrooms: 448.0
population: 338.0
households: 182.0
median_income: 1.2132
ocean_proximity: 'INLAND'
```
Output 2: `58815.45033765`

-----------------------------------

Input 3:
```
longitude: -117.96
latitude: 33.89
housing_median_age: 24.0
total_rooms: 1332.0
total_bedrooms: 252.0
population: 625.0
households: 230.0
median_income: 4.4375
ocean_proximity: '<1H OCEAN'
```
Output 3: `192575.77355635`
