from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import skrub

# %% [markdown]
# **Declare inputs**

# %%
data_path = skrub.var("data_path", value="historical_data.csv")
data_path

# %% [markdown]
# **Apply transformations:** functions, estimators, etc.

# %%
data = data_path.skb.apply_func(pd.read_csv)
data

# %%
X = data.drop(columns="target", errors="ignore").skb.mark_as_X()
y = data["target"].skb.mark_as_y()
pred = X.skb.apply(StandardScaler()).skb.apply(LogisticRegression(), y=y)
pred

# %% [markdown]
# **Validate**, inspect, etc.

# %%
pred.skb.full_report()

# %%
pred.skb.cross_validate()

# %% [markdown]
# **Create & use a "learner"** (fittable object)

# %%
learner = pred.skb.make_learner(fitted=True)
learner

# %%
learner.predict({"data_path": "new_data.csv"})
