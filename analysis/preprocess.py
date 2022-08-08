import pandas as pd

########################################################################################################################
# Variables


filepath = '../log/7_27/333_clients_4_rounds_100_a_batch_64_a_lr_0.0001.txt'
data = pd.read_csv(filepath, delimiter='INFO:root:', on_bad_lines='skip', engine='python')

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

########################################################################################################################

if __name__ == "__main__":
    # Fix cell values for all rows and columns
    data = data.drop(columns=['Unnamed: 0'])
    temp = data.columns[0]
    data = data.rename(columns={temp: 'test'})
    data = data['test'].str.split('|', expand=True)
    data.columns = ['1', '2', '3', '4', '5', '6', '7']
    data = data.dropna()
    data = data.replace('\[ Round: ', '', regex=True)
    data = data.replace('Class: ', '', regex=True)
    data = data.replace('Time: ', '', regex=True)
    data = data.replace('Loss: ', '', regex=True)
    data = data.replace('Acc: ', '', regex=True)
    data = data.replace('Conf Matrix: ', '', regex=True)
    data = data.replace(' \]', '', regex=True)
    data = data.replace('TP:', '', regex=True)
    data = data.replace('FP:', '', regex=True)
    data = data.replace('TN:', '', regex=True)
    data = data.replace('FN:', '', regex=True)
    data = data.replace('\%', '', regex=True)
    data['4'] = data['4'].replace('s', '', regex=True)
    temp = data['7'].str.split(' ', 4, expand=True)
    data = data.join(temp)
    data = data.drop(columns=['7', 0])

    data.columns = ['Round', 'Type', 'Class', 'Time', 'Loss', 'Accuracy', 'True P', 'False P', 'True N', 'False N']
    data = data.reset_index()
    data = data.drop(columns=['index'])
    data = data.astype({'Accuracy': 'double'})

    train_set = data[data['Type'] == ' Attacker Test ']
    temp = train_set[train_set['Round'] == '90 ']

    print(temp)

    print(temp['Accuracy'].mean().round(3))
