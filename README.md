# Template for model integration with CHAP 
This template provides a foundational code structure in Python for integrating models with CHAP. It includes essential components for model training and forecasting but is not designed to run immediately. For a complete, runnable example, please see to the "minimalist_example" repository.

## Running the model without CHAP integration
Before getting a new model to work as part of CHAP, it can be useful to develop and debug it while running it directly a small dataset from file. 

The example can be run in isolation (e.g. from the command line) using the file isolated_run.py:
```
python isolated_run.py  
```

This file only contains two code lines:  
* A call to a function "train", which trains a model from an input file "trainData.csv" and stores the trained model in a file "model.bin":
```python
train("input/trainData.csv", "output/model.bin")
```

* A call to a function "predict" uses the stored model to forecast future disease cases (to a file "predictions.csv") based on input data on future climate predictions (from a file futureClimateData.csv):
```python
predict("output/model.bin", "input/futureClimateData.csv", "output/predictions.csv")
```


### Training data
The example uses a minimalist input data containing rainfall, temperature and disease cases for a single region and two time points ("traindata.csv"):
```csv
time_period,rainfall,mean_temperature,disease_cases,location
2023-05,10,30,200,loc1
2023-06,2,30,100,loc1
```

### Training the model
The file "train.py" contains the code to train a model. It reads in training data from a csv file to a Pandas data frame. Then insert a code to train the model and store it to file using e.g. joblib:
```
def train(csv_fn, model_fn):
    df = pd.read_csv(csv_fn)

    # Here you train your model based on the data in df, and save it to model.bin
```
### Future climate data
A minimalist future (predicted) climate data is provided in a file "futureClimateData.csv". This file contains climate data for what is considered to be future periods (weather forecasts). It naturally contains no disease data):  
```
time_period,rainfall,mean_temperature,location
2023-07,20,20,loc1
2023-08,30,20,loc1
2023-09,30,30,loc1
```

### Generating forecasts
The file "predict.py" contains the code to forecast disease cases ahead in time based on future climate data (weather forecasts) and a previously trained model read from file. The disease forecasts are stored as a column in a csv file predictions_fn:
```
def predict(model_fn, future_climatedata_fn, predictions_fn):
    df = pd.read_csv(future_climatedata_fn)

    # Here you use your trained model to predict disease cases on the data in df, and write them out to predictions.csv
```

## Running the minimalist model as part of CHAP
To run the minimalist model in CHAP, we first define the model interface in an MLFlow-based yaml specification (in the file "MLproject", which defines :

```yaml
name: min_py_ex

adapters: {'disease_cases': 'disease_cases',
           'location': 'location',
           'time_period': 'time_period'
           'rainfall': 'rainfall',
           'mean_temperature': 'mean_temperature'}

entry_points:
  train:
    parameters:
      train_data: path
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      historic_data: path
      future_data: path
      model: str
      out_file: path
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"

```

(this is in the process of being finalised..)

