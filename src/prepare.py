import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

# Загрузка параметров
with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_size = params["prepare"]["test_size"]
random_state = params["prepare"]["random_state"]
target_column = params["prepare"]["target_column"]

# Загрузка raw data
df = pd.read_csv("data/raw/iris.csv")

# Разделяем на train и test
train, test = train_test_split(
    df,
    test_size=test_size,
    stratify=df[target_column],
    random_state=random_state
)

# Создаём папку если нет
os.makedirs("data/prepared", exist_ok=True)

# Сохраняем
train.to_csv("data/prepared/train.csv", index=False)
test.to_csv("data/prepared/test.csv", index=False)