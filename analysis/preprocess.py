import pandas as pd

########################################################################################################################
# # Variables
# filepath = '../log/Attack/active/no_overlap/575_active_attack_25.txt'
# filepath1 = '../log/Attack/active/no_overlap/577_active_attack_42.txt'
# filepath2 = '../log/Attack/active/no_overlap/579_active_attack_56.txt'
# filepath3 = '../log/Attack/active/no_overlap/572_active_attack_78.txt'
# filepath4 = '../log/Attack/active/no_overlap/576_active_attack_92.txt'

# Variables
filepath = '../log/Attack/passive/no_overlap/536_passive_attack_25.txt'
filepath1 = '../log/Attack/passive/no_overlap/560_passive_attack_42.txt'
filepath2 = '../log/Attack/passive/no_overlap/571_passive_attack_56.txt'
filepath3 = '../log/Attack/passive/no_overlap/540_passive_attack_78.txt'
filepath4 = '../log/Attack/passive/no_overlap/535_passive_attack_92.txt'

paths = [('25', filepath),
         ('42', filepath1),
         ('56', filepath2),
         ('78', filepath3),
         ('92', filepath4)]

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


def convert_dataframe(data):
    # Fix cell values for all rows and columns
    data = data.drop(columns=['Unnamed: 0'])
    temp = data.columns[0]
    data = data.rename(columns={temp: 'test'})
    data = data['test'].str.split('|', expand=True)
    data.columns = ['1', '2', '3', '4', '5', '6']
    data = data.dropna()
    data = data.replace('\[ Round: ', '', regex=True)
    data = data.replace('Class: ', '', regex=True)
    data = data.replace('Batch ', '', regex=True)
    data = data.replace('Final ', '', regex=True)
    data = data.replace('Time: ', '', regex=True)
    data = data.replace('Loss: ', '', regex=True)
    data = data.replace('Acc: ', '', regex=True)
    data = data.replace('Avg ', '', regex=True)
    data = data.replace('Conf Matrix: ', '', regex=True)
    data = data.replace(' \]', '', regex=True)
    data = data.replace('TP:', '', regex=True)
    data = data.replace('FP:', '', regex=True)
    data = data.replace('TN:', '', regex=True)
    data = data.replace('FN:', '', regex=True)
    data = data.replace('\%', '', regex=True)

    data['4'] = data['4'].replace('s', '', regex=True)
    temp = data['6'].str.split(' ', 4, expand=True)
    data = data.join(temp)
    data = data.drop(columns=['6', 0])

    data.columns = ['Round', 'Type', 'Time', 'Loss', 'Accuracy', 'True P', 'False P', 'True N', 'False N']
    data = data.reset_index()
    data = data.drop(columns=['index'])

    data = data[(data['Round'].str.contains('Totals'))]
    data = data.replace('Totals ', '', regex=True)
    data = data.replace('Total ', '', regex=True)
    data = data.replace('Attacker ', '', regex=True)
    data['Time'] = data['Time'].replace('s', '', regex=True)
    data = data.astype({'Accuracy': 'double'})
    data = data.astype({'Loss': 'double'})
    data = data.astype({'Time': 'double'})
    data = data.astype({'Round': 'int'})

    data = data.astype({'True P': 'int'})
    data = data.astype({'False P': 'int'})
    data = data.astype({'True N': 'int'})
    data = data.astype({'False N': 'int'})

    temp1 = data[(data['Type'].str.contains('Train'))]
    temp2 = data[(data['Type'].str.contains('Test'))]

    return pd.merge(temp1, temp2, on='Round')


########################################################################################################################

if __name__ == "__main__":

    with pd.ExcelWriter('../log/excel/passive_attack_no_overlap.xlsx') as writer:
        for path in paths:
            seed, file = path
            data = pd.read_csv(file, delimiter='INFO:root:', on_bad_lines='skip', engine='python', decimal=',')
            result = convert_dataframe(data)
            result.to_excel(excel_writer=writer, sheet_name=seed)
