import pandas as pd
print(pd.read_csv('big_dataset_features.csv')['label'].value_counts())