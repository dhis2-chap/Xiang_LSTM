from predict import predict
from train import train

train("input/data.csv", "output/model.bin")
predict("output/model.bin", "input/data.csv", "input/futureClimateData.csv",  "output/predictions.csv")