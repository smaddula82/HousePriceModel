import pickle
import pandas as pd
from HousePriceModel.Config import Config
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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


if __name__ == '__main__':
    predicted_house_price = predict_price()
    print(predicted_house_price)
