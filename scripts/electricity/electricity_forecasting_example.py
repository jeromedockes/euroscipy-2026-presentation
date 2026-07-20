# %% [markdown]
# # Electricity load (demand) forecasting
#
# Adapted from forecasting [course](https://github.com/probabl-ai/forecasting), thanks to Guillaume Lemaître, Olivier Grisel, Shruti Nath `@probabl`.
#
# ![](electricity_prediction.svg)
#
# ![](horizons.svg)
#
# <a href='forecasting_report/index.html' target='_blank'>full pipeline</a>

# %%
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor

import electricity_forecasting as ef

HORIZONS = (1, 12, 24)


# %% [markdown]
#
# ## Fetching the historical electricity demand
#

# %%
history_fetcher = skrub.var(
    "history_fetcher", ef.fetch_demand_history, becomes_default=True
)
demand_history = history_fetcher().skb.apply_func(ef.resample)
demand_history

# %%
ef.plot(demand_history, start="2025-01-30")

# %% [markdown]
# ## Building the query X and targets y
#
# <!-- TODO figure showing predict time & target time for 2 horizons -->

# %%
range_start = skrub.var("start", "2021-03-23")
range_end = skrub.var("end", "2025-05-31")
prediction_time = skrub.deferred(ef.time_range)(range_start, range_end)
prediction_time

# %%
X_y = prediction_time.skb.apply_func(ef.get_X_y, demand_history, HORIZONS)
X = X_y["X"].skb.mark_as_X(cv=ef.TimeSeriesSplitter())
y = X_y["y"].skb.mark_as_y()
y

# %%
X

# %% [markdown]
# ## Building predictive features

# %%
holidays_fetcher = skrub.var("holidays_fetcher", ef.fetch_holidays, becomes_default=True)
holidays_fetcher([2024, 2025])

# %%
weather_fetcher = skrub.var("weather_fetcher", ef.fetch_weather, becomes_default=True)
weather_fetcher("paris")

# %%
ef.ALL_CITIES

# %%
temperature_only = skrub.choose_bool(name="temperature_only", default=True)
cities = skrub.choose_from(
    {"all": ef.ALL_CITIES, "top_3": ["paris", "marseille", "lyon"]}, name="cities"
)

features = {
    h: X.skb.apply_func(
        ef.add_features,
        horizon=h,
        temperature_only=temperature_only,
        cities=cities,
        demand_history=demand_history,
        weather_fetcher=weather_fetcher,
        holidays_fetcher=holidays_fetcher,
    ).skb.set_name(f"feat_{h}h")
    for h in HORIZONS
}

features[12]

# %% [markdown]
# ## Adding the supervised predictors

# %%
learning_rate = skrub.choose_float(0.01, 0.7, default=0.1, log=True, name="learning_rate")
max_leaf_nodes = skrub.choose_int(3, 300, default=30, log=True, name="max_leaf_nodes")
min_samples_leaf = skrub.choose_int(2, 64, default=20, log=True, name="min_samples_leaf")
loss = skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss")

regressor = HistGradientBoostingRegressor(
    random_state=0,
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=50,
    learning_rate=learning_rate,
    max_leaf_nodes=max_leaf_nodes,
    min_samples_leaf=min_samples_leaf,
    loss=loss,
)

horizon_preds = {
    h: feat.skb.apply(regressor, y=y[f"{h}h"]).skb.set_name(f"pred_{h}h")
    for h, feat in features.items()
}
horizon_preds[12]

# %%
pred = skrub.deferred(ef.concat_horizons)(horizon_preds).skb.with_scoring(ef.neg_mape)
pred

# %%
# create the full report & open in another tab:
# pred.skb.full_report()

# %% [markdown]
# ## Assessing predictions with the default params

# %%
pred.skb.cross_validate()

# %%
split = pred.skb.train_test_split()

learner = pred.skb.make_learner().fit(split["train"])
learner_predictions = learner.predict(split["test"])
learner_predictions

# %%
ef.plot_predictions(split["X_test"], split["y_test"], learner_predictions, horizon=12)

# %% [markdown]
# ## Hyperparameter search

# %%
print(pred.skb.describe_param_grid())

# %%
search = pred.skb.make_randomized_search(
    backend="optuna",
    storage="sqlite:///optuna.db",
    study_name="randomized_search",
    # n_iter=64,
    n_iter=0,
    refit="neg_mape_average",
    fitted=True,
)

# %%
search.results_

# %%
search.plot_results(show_scores="neg_mape_average", show_times=())

# %%
learner = (
    ef.make_data_op(horizons=tuple(range(1, 25)))
    .skb.make_learner()
    .fit({"start": "2021-03-23", "end": "2025-05-31"})
)
# %%
# 1h beyond the end of our historical data
future_pred = learner.predict({"start": "2025-06-27T14:00:00"})
future_pred

# %%
ef.plot(demand_history, future_pred)
