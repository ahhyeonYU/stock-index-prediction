# packages
import pandas as pd
import requests 
import xml.etree.ElementTree as ET
import os

def collect_ecos(base_dir):
    os.chdir(base_dir)
    # get url list of ecos
    ecos_dir = './data'
    ecos_file = 'ecos_url.xlsx'
    df_ecos = pd.read_excel(f'{ecos_dir}/{ecos_file}')
    
    # urls are in the 'target_url' column
    url_list_ecos = list(df_ecos['target_url'])
    
    # get data
    ecos_df = pd.DataFrame()

    for idx, url in enumerate(url_list_ecos):
        response = requests.get(url)
        temp_df = response.json()
        # [service name][row]
        temp_df = pd.DataFrame(temp_df['StatisticSearch']['row'])
        # elimanate unnecessary columns
        temp_df = temp_df[['TIME', 'ITEM_NAME1', 'DATA_VALUE']]
        item_name = temp_df.loc[0, 'ITEM_NAME1']
        temp_df.rename(columns={'DATA_VALUE': item_name}, inplace=True)
        temp_df.drop('ITEM_NAME1', axis=1, inplace=True)
        
        if idx==0:
            ecos_df = temp_df
        else:
            ecos_df = pd.merge(ecos_df, temp_df, on='TIME', how='outer')

    ecos_df.sort_values('TIME', ascending=True, inplace=True)
    ecos_df.reset_index(drop=True, inplace=True)

    return ecos_df
