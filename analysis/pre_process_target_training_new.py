import numpy as np
import pandas as pd

########################################################################################################################
# Variables

# filepath = '../log/Target/no_overlap/484_target_training_25.txt'
# filepath1 = '../log/Target/no_overlap/539_target_training_42.txt'
# filepath2 = '../log/Target/no_overlap/481_target_training_56.txt'
# filepath3 = '../log/Target/no_overlap/482_target_training_78.txt'
# filepath4 = '../log/Target/no_overlap/483_target_training_92.txt'

filepath = '../log/Target/active/no_overlap/569_active_target_training_25.txt'
filepath1 = '../log/Target/active/no_overlap/573_active_target_training_42.txt'
filepath2 = '../log/Target/active/no_overlap/574_active_target_training_56.txt'
filepath3 = '../log/Target/active/no_overlap/563_active_target_training_78.txt'
filepath4 = '../log/Target/active/no_overlap/570_active_target_training_92.txt'

paths = [('25', filepath),
         ('42', filepath1),
         ('56', filepath2),
         ('78', filepath3),
         ('92', filepath4)]

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


def convert_dataframe_active(data):
    # Fix cell values for all rows and columns
    data = data.drop(columns=['Unnamed: 0'])
    data = data.iloc[1:]
    data = data.iloc[:-1]
    temp = data.columns[0]
    data = data.rename(columns={temp: 'test'})
    data = data['test'].str.split('|', expand=True)
    data.columns = ['1', '2', '3', '4', '5', '6']
    data = data.replace('\[ Round: ', '', regex=True)
    data = data.replace('Client: ', '', regex=True)
    data = data.replace('Batch ', '', regex=True)
    data = data.replace('Local ', '', regex=True)
    data = data.replace('Time: ', '', regex=True)
    data = data.replace('Loss: ', '', regex=True)
    data = data.replace('Tr_Acc: ', '', regex=True)
    data = data.replace(' \]', '', regex=True)
    data = data.replace('%', '', regex=True)
    data = data.iloc[26:, :]
    data = data[data['2'].str.contains('Global Model Eval started') == False]
    data = data[data['2'].str.contains('!') == False]
    data = data.astype({'1': 'int'})

    train = data[data['2'].str.contains('Train')]
    train = train.drop(columns=['3'])
    train.columns = ['1', '2', '3', '4', '5']

    train1 = train[train['1'] <= 3]
    train1['5'] = train1['5'].replace(r'^.{0,20}', '', regex=True)
    train2 = train[train['1'] > 3]
    train2 = train2[train2['1'] <= 7]
    train2['5'] = train2['5'].replace(r'^.{0,21}', '', regex=True)
    train3 = train[train['1'] > 7]
    train3 = train3[train3['1'] <= 36]
    train3['5'] = train3['5'].replace(r'^.{0,21}', '', regex=True)
    train4 = train[train['1'] == 37]
    train4['5'] = train4['5'].replace(r'^.{0,21}', '', regex=True)
    train4['5'] = train4['5'].replace('=', '', regex=True)
    train5 = train[train['1'] > 37]
    train5['5'] = train5['5'].replace(r'^.{0,22}', '', regex=True)
    train = pd.concat([train1, train2, train3, train4, train5], ignore_index=True)
    train['3'] = train['3'].replace('s', '', regex=True)
    train = train.astype({'3': 'double'})
    train = train.astype({'4': 'double'})
    train['5'] = train['5'].replace('=', '', regex=True)
    train = train.astype({'5': 'double'})
    train = train.groupby(np.arange(len(train.index)) // 4).mean()
    train.insert(1, '2', 'Train')
    train.columns = ['Round', 'Type', 'Time', 'Loss', 'Accuracy']

    eval = data[data['2'].str.contains('s')]
    eval = eval.drop(columns=['5', '6'])
    eval.insert(1, '8', 'Eval')
    eval1 = eval[eval['1'] <= 5]
    eval1['4'] = eval1['4'].replace(r'^.{0,23}', '', regex=True)
    eval2 = eval[eval['1'] > 5]
    eval2['4'] = eval2['4'].replace(r'^.{0,23}', '', regex=True)
    eval2['4'] = eval2['4'].replace('=', '', regex=True)
    eval = pd.concat([eval1, eval2], ignore_index=True)
    eval['2'] = eval['2'].replace('s', '', regex=True)
    eval = eval.astype({'2': 'double'})
    eval = eval.astype({'3': 'double'})
    eval = eval.astype({'4': 'double'})
    eval.columns = ['Round', 'Type', 'Time', 'Loss', 'Accuracy']

    active = data[data['2'].str.contains('Active')]
    active = active.drop(columns=['3', '6'])
    active.insert(4, '6', '0')
    active['4'] = active['4'].replace('s', '', regex=True)
    active['2'] = active['2'].replace(' Attack', '', regex=True)
    active = active.astype({'4': 'double'})
    active = active.astype({'5': 'double'})
    active = active.astype({'6': 'double'})
    active.columns = ['Round', 'Type_x', 'Time_x', 'Loss_x', 'Accuracy_x']

    result = pd.merge(train, eval, on='Round')
    result = pd.concat([result, active])

    return result


def convert_dataframe(data):
    # Fix cell values for all rows and columns
    data = data.drop(columns=['Unnamed: 0'])
    data = data.iloc[1:]
    data = data.iloc[:-1]
    temp = data.columns[0]
    data = data.rename(columns={temp: 'test'})
    data = data['test'].str.split('|', expand=True)
    data.columns = ['1', '2', '3', '4', '5', '6']
    data = data.replace('\[ Round: ', '', regex=True)
    data = data.replace('Client: ', '', regex=True)
    data = data.replace('Batch ', '', regex=True)
    data = data.replace('Local ', '', regex=True)
    data = data.replace('Time: ', '', regex=True)
    data = data.replace('Loss: ', '', regex=True)
    data = data.replace('Tr_Acc: ', '', regex=True)
    data = data.replace(' \]', '', regex=True)
    data = data.replace('%', '', regex=True)

    data = data[data['2'].str.contains('!') == False]
    data = data[data['2'].str.contains('Global Model Eval started') == False]
    data['6'] = data['6'].replace(r'^.{0,20}', '', regex=True)
    data = data.astype({'1': 'int'})
    temp1 = data[data['1'] <= 3]
    temp2 = data[data['1'] > 3]
    temp3 = temp2[temp2['1'] > 7]
    temp2 = temp2[temp2['1'] <= 7]
    temp4 = temp3[temp3['1'] <= 37]
    temp5 = temp3[temp3['1'] > 37]

    temp2['6'] = temp2['6'].replace(r'^.{0,1}', '', regex=True)
    temp4['6'] = temp4['6'].replace(r'^.{0,1}', '', regex=True)
    temp5['6'] = temp5['6'].replace(r'^.{0,2}', '', regex=True)
    data = pd.concat([temp1, temp2, temp4, temp5])

    temp1 = data[data['2'].str.contains('s')]
    data = data[data['2'].str.contains('s') == False]
    temp1.insert(1, '8', 'Global')
    temp1.insert(1, '7', 'Eval')
    temp1 = temp1.drop(columns=['5', '6'])

    temp1.columns = ['1', '2', '3', '4', '5', '6']
    temp2 = temp1[temp1['1'] <= 6]
    temp3 = temp1[temp1['1'] > 6]
    temp2['6'] = temp2['6'].replace(r'^.{0,23}', '', regex=True)
    temp3['6'] = temp3['6'].replace(r'^.{0,24}', '', regex=True)
    temp1 = pd.concat([temp2, temp3])

    temp1.columns = ['Round', 'Type', 'ID', 'Time', 'Loss', 'Accuracy']
    temp1['Time'] = temp1['Time'].replace('s', '', regex=True)
    temp1 = temp1.astype({'Loss': 'double'})
    temp1 = temp1.astype({'Time': 'double'})

    data.columns = ['Round', 'Type', 'ID', 'Time', 'Loss', 'Accuracy']
    data['Time'] = data['Time'].replace('s', '', regex=True)
    data = data.replace('=', '', regex=True)

    data = data.astype({'Accuracy': 'double'})
    data = data.astype({'Loss': 'double'})
    data = data.astype({'Time': 'double'})
    data = data.astype({'Round': 'int'})
    data = data.groupby(np.arange(len(data.index)) // 4).mean()

    result = pd.merge(data, temp1, on='Round')
    result = result.drop(columns=['ID'])
    result = result.replace('=', '', regex=True)
    result = result.astype({'Accuracy_y': 'double'})

    return result


########################################################################################################################


if __name__ == "__main__":
    # data = pd.read_csv(filepath, delimiter='INFO:root:', on_bad_lines='skip', engine='python', decimal=',')
    # data = convert_dataframe_active(data)
    # print(data)

    with pd.ExcelWriter('../log/excel/target_training_active_no_overlap.xlsx') as writer:
        for path in paths:
            seed, file = path
            data = pd.read_csv(file, delimiter='INFO:root:', on_bad_lines='skip', engine='python', decimal=',')
            result = convert_dataframe_active(data)
            result.to_excel(excel_writer=writer, sheet_name=seed)
