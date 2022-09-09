import pandas as pd

########################################################################################################################
# Variables


filepath = '../log/Target/Target_train_client_4/298_Target_training_clients_4_rounds_300_batch_64_lr_0' \
           '.01_overlap_yes_overlapsize_30000.txt '
data = pd.read_csv(filepath, delimiter='INFO:root:', on_bad_lines='skip', engine='python')

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

########################################################################################################################

if __name__ == "__main__":
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
    data = data.replace('Time: ', '', regex=True)
    data = data.replace('Loss: ', '', regex=True)
    data = data.replace('Tr_Acc: ', '', regex=True)
    data = data.replace(' \]', '', regex=True)

    data['4'] = data['4'].replace('s', '', regex=True)
    temp = data['6'].str.split(' ', 4, expand=True)
    data = data.join(temp)
    data = data.drop(columns=['6', 0])

    data.columns = ['Round', 'Type', 'Time', 'Loss', 'Accuracy', "temp1", "temp2"]
    data = data[data['Time'].notna()]
    data = data[data['Loss'].str.contains('Accuracy')]
    print(data)