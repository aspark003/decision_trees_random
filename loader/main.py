import pandas as pd
import shutil
import os
os.makedirs('', exist_ok=True)


shutil.unpack_archive(r'c:/Users/anton/kaggle.zip', extract_dir='', format='zip')
df = pd.read_csv(r'housing.csv')
df.to_csv('c:/extra_tree/loader/tree.csv',index=False)

print(df.info())

