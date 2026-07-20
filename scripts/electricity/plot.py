import electricity_forecasting as ef

load_history = ef.fetch_load_history()
# fig = ef.plot(load_history, start="2025-01-30")
learner = (
    ef.make_data_op().skb.make_learner().fit({"start": "2021-03-23", "end": "2025-05-31"})
)
future_pred = learner.predict({"start": "2025-06-27T14:00:00"})
fig = ef.plot(load_history, future_pred)
fig.show(renderer='browser')
fig.update_layout(width=1200)
fig.write_image("electricity_prediction.svg")
