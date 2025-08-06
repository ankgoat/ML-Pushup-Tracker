import pandas as pd

df = pd.read_csv('landmark_data_big/labels.csv', header=None, names=['frame_path', 'raw_label'])

def extract_label(row):
    path = str(row['frame_path']).replace('/', '\\')
    parts = path.split('\\')
    if len(parts) > 1:
        folder = parts[1]
        if folder.startswith('good_'):
            return 'good'
        elif folder.startswith('bad_'):
            return 'bad'
    return 'unknown'

df['label'] = df.apply(extract_label, axis=1)
print(df['label'].value_counts())
# Only keep 'good' and 'bad'
df = df[df['label'].isin(['good', 'bad'])]
df[['frame_path', 'label']].to_csv('landmark_data_big/labels_clean.csv', index=False)
