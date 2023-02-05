import pickle
import pandas as pd
from Config import Config
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from SyntheticData import getsyntheticdata
import numpy as np

init_object = Config()  # initialising the object

def predict_price():
    loaded_model = pickle.load(open(init_object.model_path + init_object.model_file, 'rb'))
    csv_file_name = init_object.data_path + init_object.test_file

    test = pd.read_csv(csv_file_name, encoding="UTF-8")

    test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id'], axis=1, inplace=True)
    categorical_features = test.select_dtypes(include="object").columns
    integer_features = test.select_dtypes(exclude="object").columns

    for c in categorical_features:
        lbl = LabelEncoder()
        lbl.fit(list(test[c].values))
        test[c] = lbl.transform(list(test[c].values))

    integer_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ints', integer_transformer, integer_features),
            ('cat', categorical_transformer, categorical_features)])

    X_test = preprocessor.fit_transform(test)

    predicted_house_price = loaded_model.predict(X_test)

    return predicted_house_price

def predict_synthetic_data_price(data):
    loaded_model = pickle.load(open(init_object.model_path + init_object.model_file, 'rb'))

    predicted_house_price = loaded_model.predict(data)
    return predicted_house_price

def convert_synthetic_data(data):
    df = pd.DataFrame.from_dict(data, orient='index')
    df=df.T
    # print("printing dataframe: ", df)
    numerical_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
                          'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                          'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                          'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                          'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                          'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                          'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
                          'MoSold', 'YrSold']
    categorical_features = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
                            'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                            'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                            'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                            'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
    df[numerical_features]=df[numerical_features].astype('float64')

    csv_file_name = init_object.data_path + init_object.test_file

    test = pd.read_csv(csv_file_name, encoding="UTF-8")

    test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id'], axis=1, inplace=True)
    # print("Shape of test before append: ",test.shape)
    # print("Last row before append: ",test.tail(1))
    test=test.append(df,ignore_index=True)
    # print("Shape of test after append: ", test.shape)
    # print("Last row after append: ",test.tail(1))
    for c in categorical_features:
        lbl = LabelEncoder()
        lbl.fit(list(test[c].values))
        test[c] = lbl.transform(list(test[c].values))
    integer_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ints', integer_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    X = preprocessor.fit_transform(test)
    # print("Shape of test after preprocessor: ", X.shape)
    # print(" After preprocessor: ",X)
    # print("Last row: ",X[-1,:])

    return X[-1,:]



if __name__ == '__main__':
    predicted_house_price = predict_price()
    print("Model Predicted Price: ",predicted_house_price)
    while True:
        sflg=input("Do you want to create synthetic data? Y or N")
        if sflg=="Y":
            sdata=getsyntheticdata()
            print(sdata)
            converted_Data=convert_synthetic_data(sdata)
            reshaped_house_data = np.array(converted_Data).reshape(-1, 75)
            predicted_house_price = predict_synthetic_data_price(reshaped_house_data)
            print(predicted_house_price)
        else:
            exit()
