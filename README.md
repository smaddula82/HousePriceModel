# HousePriceModel
The model contains following files  

properties.yml which contains the basic properties. 

Config.py to load the properties.yml. 

ModelTrainer.py to train the model using the GradientBoostingRegressor algorithm. This will also dump the model in models folder. 

ModelPredictor.py to predict the house price using the model dumped from ModelTrainer.py. This will also run in interactive session to create the synthetic data and predict the price. 

SyntheticData.py will create the data using the training data available. 
For docker image please go through DockerImage.docx
