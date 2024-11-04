import pandas as pd

def pippo(path):
    columns = ['sex',
                'length',
                'diameter',
                'height',
                'whole_weight',
                'shucked_weight',
                'viscera_weight',
                'shell_weight',
                'rings'
                ]
    df = pd.read_csv(path, names=columns, header=None)
    print(df.head())
    print(df.describe())
