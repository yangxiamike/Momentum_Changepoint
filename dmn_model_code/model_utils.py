import numpy as np
from sklearn.preprocessing import StandardScaler

def split_by_date(df, date_breakpoints):

    data_sets = []
    for idx, start in enumerate(date_breakpoints):
        if idx == len(date_breakpoints) - 1:
            break
        end = date_breakpoints[idx + 1]
        data_sets.append(df[(start <= df['date']) & (df['date'] < end)])
    data_sets.append(df[df['dates'] >= end])
    return data_sets

def split_by_category(df, start_date, end_date):
    train_sets = []
    test_sets = []
    df.set_index(['ticker', 'date'], inplace = True, drop = False)
    df.sort_index(inplace = True)

    for asset, df_piece in df.groupby(level = 0):
        train_data = df_piece[df_piece['date'] < start_date]
        test_data = df_piece[(df_piece['date'] >= start_date) & (df_piece['date'] < end_date)]
        train_data.dropna(inplace = True)
        test_data.dropna(inplace = True)
        
        train_data.drop(['ticker', 'date'], inplace = True, axis = 1)
        test_data.drop(['ticker', 'date'], inplace = True, axis = 1)

        scaler = StandardScaler()
        train_data.iloc[:, :-2] = scaler.fit_transform(train_data.iloc[:, :-2])
        test_data.iloc[:, :-2] = scaler.transform(test_data.iloc[:, :-2])

        train_sets.append(train_data)
        test_sets.append(test_data)

        # todo 删除不必要的列

    return train_sets, test_sets

