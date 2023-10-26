from pandas import read_csv, DataFrame, concat
from catboost import CatBoostRegressor
import pickle as pkl

def main(path:str) -> None:
    df = read_csv(path)
    data = preprocessing(df)
    create_data_pred(data)    

def create_data_pred(data:DataFrame) -> None:
    X_test = data.drop(['target'], axis=1)
    model = pkl.load(open('cat_optuna.pkl', 'rb'))
    y_pred = model.predict(X_test)
    data_pred = DataFrame(y_pred, index=data.index, columns = ['y_pred'])
    data_pred.to_csv('./../data/data_pred.csv')

def preprocessing(data:DataFrame) -> DataFrame:
    data = new_data(data)
    data['diff'] = data['target'].diff()
    data = lagged(data, [1, 7], 'target')
    data = lagged(data, [1, 7], 'diff')
    return data.drop(['diff'], axis=1).set_index('date').dropna()  

def new_data(data:DataFrame) -> DataFrame:
    data["target"] = data.groupby(['date'])[['target']].transform(('sum'))
    data["temp_pred"] = data.groupby(['date'])[['temp_pred']].transform(('mean'))
    data = data[['date', 'target', 'temp_pred']]
    return data.drop_duplicates()

def lagged(data:DataFrame, lags:list, name_column:str) -> DataFrame:
    for i in lags:
        data[f'{name_column}_lag{i}'] = data[name_column].shift(periods=i)
    return data

path = input('Введите путь: ')
main(path)