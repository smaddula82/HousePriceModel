from faker import Faker
from Config import Config
import pandas as pd

init_object = Config()

fake = Faker()

def get_house_prediction_data(data,norec):

    list1=[]
    for _ in range(norec):
        dict1={}
        for row in data.columns:
            print(row,": ",data[row].unique().tolist())
            cleanedList = [x for x in data[row].unique().tolist() if x == x]
            print(row,": ",cleanedList)
            dict1[row]=fake.random.choice(cleanedList)
        list1.append(dict1)

    return list1
def getsyntheticdata():
    csv_file_name = init_object.data_path + init_object.data_file
    print(csv_file_name)
    train = pd.read_csv(csv_file_name, encoding="UTF-8")
    train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id', 'SalePrice'], axis=1, inplace=True)
    dict1 = {}
    for row in train.columns:
        cleanedList = [x for x in train[row].unique().tolist() if x == x]
        dict1[row] = fake.random.choice(cleanedList)
    return dict1

if __name__ == '__main__':
    csv_file_name = init_object.data_path + init_object.data_file
    print(csv_file_name)
    train = pd.read_csv(csv_file_name, encoding="UTF-8")
    train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id','SalePrice'], axis=1, inplace=True)
    # print(train.columns)
    # print(get_house_prediction_data(train,50))
    list2=get_house_prediction_data(train,2)
    df=pd.DataFrame.from_records(list2)
    print(df.head())

