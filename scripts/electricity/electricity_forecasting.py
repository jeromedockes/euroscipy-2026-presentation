import datetime
from pathlib import Path

import polars as pl
import skrub
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import HistGradientBoostingRegressor
from polars import selectors as cs
import plotly.graph_objects as go
import holidays

skrub.set_config(table_report_n_rows=5, data_ops_open_graph_dropdown=True)

ALL_CITIES = (
    "paris",
    "lyon",
    "marseille",
    "toulouse",
    "lille",
    "limoges",
    "nantes",
    "strasbourg",
    "brest",
    "bayonne",
)
TRAIN_TEST_GAP_DAYS = 7

ENV = {"start": "2021-03-23", "end": "2025-05-31"}
NEW_DATE = "2025-06-27T14:00:00"


def data_dir():
    return Path("datasets")


def time_range(start, end=None):
    if end is None:
        end = start
    if isinstance(start, str):
        start = datetime.datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.datetime.fromisoformat(end)
    return pl.DataFrame().with_columns(
        pl.datetime_range(
            start=start,
            end=end,
            time_zone="UTC",
            interval="1h",
        )
        .dt.truncate("1h")
        .alias("time"),
    )


def resample(demand_history):
    averaged = demand_history.group_by(pl.col("time").dt.truncate("1h")).agg(
        pl.col("load_MW").mean()
    )
    all_times = averaged["time"]
    return time_range(
        all_times.min(), all_times.max() + datetime.timedelta(hours=48)
    ).join(averaged, on="time", how="left", maintain_order="left")


def fetch_demand_history():
    return (
        pl.read_csv(data_dir() / "Total Load - Day Ahead*.csv", null_values=["N/A", "-"])
        .drop_nulls()
        .select(
            pl.col("Time (UTC)")
            .str.split(by=" - ")
            .list.first()
            .str.to_datetime("%d.%m.%Y %H:%M", time_zone="UTC")
            .alias("time"),
            pl.col("Actual Total Load [MW] - BZN|FR").cast(pl.Float32).alias("load_MW"),
        )
    )


def _split_indices(X, test_start_date, test_length_days):
    train = (
        X.with_row_index()
        .filter(
            pl.col("prediction_time")
            < test_start_date - datetime.timedelta(TRAIN_TEST_GAP_DAYS)
        )["index"]
        .to_numpy()
    )
    test = (
        X.with_row_index()
        .filter(
            (pl.col("prediction_time") >= test_start_date)
            & (
                pl.col("prediction_time")
                < test_start_date + datetime.timedelta(days=test_length_days)
            )
        )["index"]
        .to_numpy()
    )
    return train, test


class TimeSeriesSplitter:
    def split(self, X, y=None, groups=None):
        min_train_days = 365 * 2
        test_length_days = 24 * 7  # 24 weeks
        test_start_dates = pl.date_range(
            X["prediction_time"].min()
            + datetime.timedelta(days=min_train_days + TRAIN_TEST_GAP_DAYS),
            X["prediction_time"].max(),
            interval=datetime.timedelta(days=test_length_days),
            closed="left",
            eager=True,
        )
        for test_start in test_start_dates:
            train, test = _split_indices(X, test_start, test_length_days=test_length_days)
            if len(train) and len(test):
                yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        return len(list(self.split(X, y)))


def get_X_y(prediction_time, demand_history, horizons, mode=skrub.eval_mode()):
    prediction_time = prediction_time.rename({"time": "prediction_time"})
    if mode in ("fit", "fit_transform", "preview"):
        demand = demand_history.select(
            pl.col("time"),
            *[pl.col("load_MW").shift(-h).alias(f"{h}h") for h in horizons],
        ).drop_nulls()
        X_y = prediction_time.join(
            demand,
            left_on="prediction_time",
            right_on="time",
            how="inner",
            maintain_order="left",
        )
        return {
            "X": X_y.select(pl.col("prediction_time")),
            "y": X_y.drop("prediction_time"),
        }
    else:
        return {"X": prediction_time}


def add_target_time(df, horizon):
    return df.with_columns(
        (pl.col("prediction_time") + pl.duration(hours=horizon)).alias("target_time")
    )


def add_lagged_features(df, demand_history, horizon):
    assert horizon <= 24
    lags = (
        pl.col("load_MW").shift(lag).alias(f"lag_{lag}")
        for lag in list(range(horizon, 24)) + [24, 24 * 2, 24 * 7]
    )

    rolling_lags = sorted(set((horizon, 24)))
    rolling_widths = (6, 24, 24 * 7)

    def rolling(e, name):
        return [
            e.rolling(
                index_column="time", period=f"{width}h", offset=f"{-width -lag}h"
            ).alias(f"lag_{lag}_width_{width}_{name}")
            for lag in rolling_lags
            for width in rolling_widths
        ]

    medians = rolling(pl.col("load_MW").median(), "median")
    iqr = rolling(
        (pl.col("load_MW").quantile(0.75) - pl.col("load_MW").quantile(0.25)), "iqr"
    )
    features = demand_history.select(pl.col("time"), *lags, *medians, *iqr)
    return df.join(
        features,
        left_on="target_time",
        right_on="time",
        how="left",
        maintain_order="left",
    )


def fetch_weather(city):
    return pl.read_parquet(data_dir() / f"weather_{city}.parquet")


def add_weather(
    df,
    cities="all",
    temperature_only=True,
    weather_fetcher=fetch_weather,
):
    if isinstance(cities, str):
        assert cities == "all"
        cities = ALL_CITIES
    with_weather = df
    for city in cities:
        with_weather = with_weather.join(
            weather_fetcher(city)
            .with_columns(pl.col("time").dt.cast_time_unit("us"))
            .select(
                (pl.col("time"), cs.matches(".*temperature.*"))
                if temperature_only
                else pl.all()
            )
            .select(
                pl.col("time"),
                (~cs.by_name("time")).as_expr().name.map(f"weather_{{}}_{city}".format),
            ),
            left_on="target_time",
            right_on="time",
            how="left",
            maintain_order="left",
        )
    return with_weather


def fetch_holidays(years):
    return holidays.country_holidays("FR", years=years)


def add_calendar_and_holidays(df, holidays_fetcher=fetch_holidays):
    fr_time = pl.col("target_time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = df.select(fr_time.dt.year().min()).item()
    fr_year_max = df.select(fr_time.dt.year().max()).item()
    holidays_fr = fetch_holidays(years=range(fr_year_min, fr_year_max + 1))
    return df.with_columns(
        fr_time.dt.hour().alias("cal_hour_of_day"),
        fr_time.dt.weekday().alias("cal_day_of_week"),
        fr_time.dt.ordinal_day().alias("cal_day_of_year"),
        fr_time.dt.year().alias("cal_year"),
        fr_time.dt.date().is_in(holidays_fr.keys()).alias("cal_is_holiday"),
    )


def add_features(
    df,
    *,
    horizon,
    temperature_only,
    cities,
    demand_history,
    weather_fetcher,
    holidays_fetcher,
    drop_time=True,
):
    df = add_target_time(df, horizon=horizon)
    df = add_weather(
        df,
        temperature_only=temperature_only,
        cities=cities,
        weather_fetcher=weather_fetcher,
    )
    df = add_calendar_and_holidays(df, holidays_fetcher)
    df = add_lagged_features(df, demand_history=demand_history, horizon=horizon)
    if drop_time:
        df = df.drop(["prediction_time", "target_time"])
    return df


def concat_horizons(all_pred):
    return pl.DataFrame({f"{h}h": v for h, v in all_pred.items()})


def neg_mape_from_predictions(y_true, y_pred):
    average = mean_absolute_percentage_error(y_true, y_pred)
    detail = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    return {"neg_mape_average": -average} | {
        f"neg_mape_{c}": -float(s) for c, s in zip(y_true.columns, detail)
    }


def neg_mape(estimator, X, y):
    return neg_mape_from_predictions(y, estimator.predict(X))


def post_process(pred, X, range_end):
    if range_end is not None:
        return pred
    pred_time = X["prediction_time"].to_list()[0]
    time = [
        pred_time + datetime.timedelta(hours=int(c.removesuffix("h")))
        for c in pred.columns
    ]
    return pl.DataFrame({"time": time, "load_MW": pred.row(0)})


def make_data_op(horizons=tuple(range(1, 15))):
    history_fetcher = skrub.var(
        "history_fetcher", fetch_demand_history, becomes_default=True
    )
    demand_history = history_fetcher().skb.apply_func(resample)
    range_start = skrub.var("start")
    range_end = skrub.var("end", None, becomes_default=True)
    prediction_time = skrub.deferred(time_range)(range_start, range_end)
    X_y = prediction_time.skb.apply_func(get_X_y, demand_history, horizons)
    X = X_y["X"].skb.mark_as_X(cv=TimeSeriesSplitter())
    y = X_y["y"].skb.mark_as_y()
    weather_fetcher = skrub.var("weather_fetcher", fetch_weather, becomes_default=True)
    temperature_only = skrub.choose_bool(name="temperature_only", default=True)
    cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")
    holidays_fetcher = skrub.var("holidays_fetcher", fetch_holidays, becomes_default=True)
    features = {
        h: X.skb.apply_func(
            add_features,
            horizon=h,
            temperature_only=temperature_only,
            cities=cities,
            demand_history=demand_history,
            weather_fetcher=weather_fetcher,
            holidays_fetcher=holidays_fetcher,
        ).skb.set_name(f"feat_{h}h")
        for h in horizons
    }
    learning_rate = skrub.choose_float(
        0.01, 0.7, default=0.1, log=True, name="learning_rate"
    )
    max_leaf_nodes = skrub.choose_int(3, 300, default=30, log=True, name="max_leaf_nodes")
    min_samples_leaf = skrub.choose_int(
        2, 64, default=20, log=True, name="min_samples_leaf"
    )
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
    pred = (
        skrub.deferred(concat_horizons)(horizon_preds)
        .skb.apply_func(post_process, X, range_end)
        .skb.with_scoring(neg_mape)
    )
    return pred


def plot(demand_history, future_pred=None, start=None, end=None):
    if isinstance(demand_history, skrub.DataOp):
        demand_history = demand_history.skb.preview()
    if start is None and future_pred is not None:
        start = (demand_history["time"].max() - datetime.timedelta(days=8)).isoformat()
    if start is None:
        start = demand_history["time"].min()
    else:
        start = datetime.datetime.fromisoformat(start).astimezone(datetime.UTC)
    demand_history = demand_history.filter(pl.col("time") >= start)
    if end is None:
        end = (
            demand_history["time"].max()
            if future_pred is None
            else future_pred["time"].max()
        )
    else:
        end = datetime.datetime.fromisoformat(end).astimezone(datetime.UTC)
    demand_history = demand_history.filter(pl.col("time") < end)
    fig = go.Figure()
    kwargs = {} if future_pred is None else {"line": {"dash": "dash", "color": "gray"}}
    fig.add_trace(
        go.Scatter(
            x=demand_history["time"],
            y=demand_history["load_MW"],
            mode="lines+markers",
            name="Load (MW)",
            **kwargs,
        )
    )
    if future_pred is not None:
        fig.add_trace(
            go.Scatter(
                x=future_pred["time"],
                y=future_pred["load_MW"],
                mode="lines+markers",
                name="Predicted load (MW)",
            )
        )
    fig.update_xaxes(
        range=(start + datetime.timedelta(hours=1), end + datetime.timedelta(hours=12)),
        showgrid=False,
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(showgrid=False, showline=True, linecolor="black")
    fig.update_layout(
        height=700 if future_pred is None else 600,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_predictions(X_test, y_test, y_pred, horizon):
    fig = go.Figure()
    target_time = X_test["prediction_time"] + datetime.timedelta(hours=horizon)
    fig.add_trace(
        go.Scatter(
            x=target_time,
            y=y_test[f"{horizon}h"],
            mode="lines+markers",
            line={"dash": "dash"},
            name="true_load_MW",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=target_time,
            y=y_pred[f"{horizon}h"],
            mode="lines",
            name=f"{horizon}h prediction",
        )
    )
    start, end = target_time.min(), target_time.max()
    fig.update_xaxes(
        range=(start + datetime.timedelta(hours=1), end + datetime.timedelta(hours=12)),
        showgrid=False,
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(showgrid=False, showline=True, linecolor="black")
    fig.update_layout(
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
