import pandas as pd
for fname in ['train.csv', 'val.csv', 'test.csv']:
    df = pd.read_csv(fname)
    print(f"{fname}: {df['label'].value_counts()}")