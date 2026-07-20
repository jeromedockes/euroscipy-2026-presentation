import skrub
from skrub import selectors as s

X = skrub.var("X")
numeric = X.skb.select(s.numeric()).skb.set_name('numeric').skb.apply(skrub.SquashingScaler())
string = X.skb.select(s.string()).skb.set_name('string').skb.apply(skrub.StringEncoder())
datetime = X.skb.select(s.any_date()).skb.set_name('datetime').skb.apply(skrub.DatetimeEncoder())
vectorized = numeric.skb.concat([string, datetime], axis=1).skb.set_name("vectorized")

# %%
with open("table_vectorizer_graph.svg", "wb") as f:
    f.write(vectorized.skb.draw_graph().svg)

# %%

# import pandas as pd

# df = pd.DataFrame(
#     {
#         "event": ["concert", "football game"],
#         "date": pd.to_datetime(["2025-02-03", "2025-11-05"]),
#         "tickets_sold": [1500, 2300],
#     }
# )
# vectorized.skb.full_report(
#     {"X": df}, open=False, output_dir="table_vectorizer_report", overwrite=True
# )
