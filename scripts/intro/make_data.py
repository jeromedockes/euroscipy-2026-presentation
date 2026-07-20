from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

X, y = make_classification(n_samples=200, n_features=5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=0)
columns="A B C D E".split()
df_train = pd.DataFrame(X_train, columns=columns ).assign(target=y_train)
df_test = pd.DataFrame(X_test, columns=columns)

df_train.to_csv("historical_data.csv", index=False)
df_test.to_csv("new_data.csv", index=False)
