import numpy as np
import pandas as pd

########################################################################################################################
# Variables

# # Variables
# filepath = '../log/Target/Target_train_client_4/299_Target_training_clients_4_rounds_300_batch_64_lr_0' \
#            '.01_overlap_yes_overlapsize_30000.txt'
# filepath1 = '../log/Target/Target_train_client_4/295_Target_training_clients_4_rounds_300_batch_64_lr_0' \
#            '.01_overlap_yes_overlapsize_30000.txt'
# filepath2 = '../log/Target/Target_train_client_4/296_Target_training_clients_4_rounds_300_batch_64_lr_0' \
#            '.01_overlap_yes_overlapsize_30000.txt'
# filepath3 = '../log/Target/Target_train_client_4/297_Target_training_clients_4_rounds_300_batch_64_lr_0' \
#            '.01_overlap_yes_overlapsize_30000.txt'
# filepath4 = '../log/Target/Target_train_client_4/298_Target_training_clients_4_rounds_300_batch_64_lr_0' \
#            '.01_overlap_yes_overlapsize_30000.txt'


# Variables
filepath = '../log/Target/no_overlap/484_target_training_25.txt'
filepath1 = '../log/Target/no_overlap/539_target_training_42.txt'
filepath2 = '../log/Target/no_overlap/481_target_training_56.txt'
filepath3 = '../log/Target/no_overlap/482_target_training_78.txt'
filepath4 = '../log/Target/no_overlap/483_target_training_92.txt'

paths = [('25', filepath),
         ('42', filepath1),
         ('56', filepath2),
         ('78', filepath3),
         ('92', filepath4)]

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


########################################################################################################################

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
    temp1 = data[data['1'] <= 2]
    temp2 = data[data['1'] > 2]
    temp3 = temp2[temp2['1'] > 6]
    temp2 = temp2[temp2['1'] <= 6]
    temp4 = temp3[temp3['1'] <= 21]
    temp5 = temp3[temp3['1'] > 21]

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
    temp2 = temp1[temp1['1'] <= 5]
    temp3 = temp1[temp1['1'] > 5]
    temp2['6'] = temp2['6'].replace(r'^.{0,23}', '', regex=True)
    temp3['6'] = temp3['6'].replace(r'^.{0,24}', '', regex=True)
    temp1 = pd.concat([temp2, temp3])

    temp1.columns = ['Round', 'Type', 'ID', 'Time', 'Loss', 'Accuracy']
    temp1['Time'] = temp1['Time'].replace('s', '', regex=True)
    temp1 = temp1.astype({'Loss': 'double'})
    temp1 = temp1.astype({'Time': 'double'})

    data.columns = ['Round', 'Type', 'ID', 'Time', 'Loss', 'Accuracy']
    data['Time'] = data['Time'].replace('s', '', regex=True)
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


if __name__ == "__main__":

    with pd.ExcelWriter('../log/excel/target_training_no_overlap.xlsx') as writer:
        for path in paths:
            seed, file = path
            data = pd.read_csv(file, delimiter='INFO:root:', on_bad_lines='skip', engine='python', decimal=',')
            result = convert_dataframe(data)
            result.to_excel(excel_writer=writer, sheet_name=seed)
