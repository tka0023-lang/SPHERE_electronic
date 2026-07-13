import pandas as pd

# Прочитать файл
df = pd.read_parquet('results.parquet')

# Посмотреть первые строки
print(df.head())

# Получить информацию о данных
print(df.info())

# Базовую статистику
print(df.describe())


print(df['Rc_snow'].max())
print(df['COSpl'].max())