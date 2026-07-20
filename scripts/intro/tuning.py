from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import skrub

# %% [markdown]
# **Declare DataOp**

# %%
data_path = skrub.var("data_path", value="historical_data.csv")
data = data_path.skb.apply_func(pd.read_csv)
X = data.drop(columns="target", errors="ignore").skb.mark_as_X()
y = data["target"].skb.mark_as_y()

# %%
squashing = skrub.SquashingScaler(
    quantile_range=skrub.choose_from([(25.0, 75.0), (10.0, 90.0)], name="q_range")
)
scaler = skrub.choose_from(
    {"standard": StandardScaler(), "squashing": squashing}, name="scaler"
)
features = X.skb.apply(scaler)

# %%
logreg = LogisticRegression(C=skrub.choose_float(0.01, 10.0, log=True, name="C"))
ridge = RidgeClassifier(alpha=skrub.choose_float(0.02, 20.0, log=True, name="alpha"))
classifier = skrub.choose_from({"logreg": logreg, "ridge": ridge}, name="classifier")
pred = features.skb.apply(classifier, y=y)
pred

# %% [markdown]
# **Hyperparameter search**

# %%
print(pred.skb.describe_param_grid())

# %%
search = pred.skb.make_randomized_search(fitted=True, n_iter=16, random_state=0)

# %%
search.results_

# %%
search.plot_results()

# %%
search.best_learner_

# %%
search.best_learner_.describe_params()
