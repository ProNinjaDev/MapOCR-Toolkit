import pandas as pd

df = pd.read_csv('data\dataset_LABELED.csv', sep=';')
print(df['label'].value_counts().sort_values(ascending=False))
print('\nВсего примеров:', len(df))