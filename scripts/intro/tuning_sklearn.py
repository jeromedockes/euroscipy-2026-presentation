from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from skrub import SquashingScaler
from scipy import stats
import pandas as pd
import skrub

# %% [markdown]
# **Declare DataOp**

# %%
data_path = skrub.var("data_path", value="historical_data.csv")
data = pd.read_csv("historical_data.csv")
X = data.drop(columns="target")
y = data["target"]

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ]
)
param_distributions = [
    {
        "scaler": [StandardScaler()],
        "classifier": [LogisticRegression()],
        "classifier__C": stats.loguniform(0.01, 10.0),
    },
    {
        "scaler": [SquashingScaler()],
        "scaler__quantile_range": [(25.0, 75.0), (10.0, 90.0)],
        "classifier": [LogisticRegression()],
        "classifier__C": stats.loguniform(0.01, 10.0),
    },
    {
        "scaler": [StandardScaler()],
        "classifier": [RidgeClassifier()],
        "classifier__alpha": stats.loguniform(0.02, 20.0),
    },
    {
        "scaler": [SquashingScaler()],
        "scaler__quantile_range": [(25.0, 75.0), (10.0, 90.0)],
        "classifier": [RidgeClassifier()],
        "classifier__alpha": stats.loguniform(0.02, 20.0),
    },
]
search = RandomizedSearchCV(pipeline, param_distributions).fit(X, y)

# %%
pd.DataFrame(search.cv_results_)

# %%
search.best_params_
