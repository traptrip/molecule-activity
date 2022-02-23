import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("base/train.csv").drop("Unnamed: 0", axis=1)
df["Active"] = df.Active.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df[["Smiles"]], df["Active"], test_size=0.2, random_state=42
)

X_train["Active"] = y_train
X_test["Active"] = y_test

X_train.to_csv("automl/train.csv", index=False)
X_test.to_csv("automl/val.csv", index=False)

test = pd.read_csv("base/test.csv").drop("Unnamed: 0", axis=1)
test.to_csv("automl/test.csv", index=False)
