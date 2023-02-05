import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from HousePriceModel.Config import Config

init_object= Config() #initialising the object


def file_reader(): #reading the test data
    csv_file_name=init_object.data_path+init_object.data_file
    print(csv_file_name)
    train = pd.read_csv(csv_file_name, encoding="UTF-8")
    train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id'], axis=1, inplace=True)
    X = train.drop(['SalePrice'], axis=1)  # Features
    y = train[['SalePrice']].values.ravel()  # Target variable
    return X,y

def model_training(X,y): #model Training
    categorical_features = X.select_dtypes(include="object").columns
    integer_features = X.select_dtypes(exclude="object").columns
    print("categorical features: ",categorical_features)
    print("numerical features: ",integer_features)

    for c in categorical_features:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))

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

    X = preprocessor.fit_transform(X)
    X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.20, random_state=42,shuffle=False)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    GBoost.fit(X_train, y_train)
    csv_file_name = init_object.data_path + init_object.test_file

    test = pd.read_csv(csv_file_name,encoding="UTF-8")

    test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id'], axis=1, inplace=True)

    for c in categorical_features:
        lbl = LabelEncoder()
        lbl.fit(list(test[c].values))
        test[c] = lbl.transform(list(test[c].values))

    X_test = preprocessor.fit_transform(test)

    prediction = GBoost.predict(X_test)  # making the prediction of the test data

    print(prediction)
    model_obj = init_object.model_path + init_object.model_file
    pickle.dump(GBoost, open(model_obj, 'wb'))
    print("Model Dumped")


if __name__=='__main__':
    x_list, y_list = file_reader()
    model_training(x_list, y_list)
