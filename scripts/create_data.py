import os
import pandas as pd

path = 'cluster_op_old'

def make_df(folder_name):

    file_list = []
    for file_name in sorted(os.listdir(os.path.join(os.getcwd(), path, folder_name))):
        file_list.append([file_name, folder_name])
    df = pd.DataFrame(file_list, columns=['files', 'labels'])
    return df

df_front = make_df('front')
df_top = make_df('top')
df_left = make_df('left')
df_right = make_df('right')

datafr = pd.concat([df_front, df_top, df_left, df_right], ignore_index=True, sort=True)
datafr.to_csv(os.path.join(os.getcwd(), path, 'datafile.csv'))