from pathlib import Path
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

skrub.set_config(table_report_n_rows=5)

dataset = skrub.datasets.fetch_credit_fraud(split="train")

data_source = skrub.var(
    "data_source", {"orders": dataset.baskets, "items": dataset.products}
)


def load_orders(data_source, order_ids=None):
    if order_ids is None:
        return data_source["orders"]
    return data_source["orders"].set_index("ID").loc[order_ids].reset_index(drop=False)


query = skrub.var("order_ids", dataset["baskets"]["ID"])
orders = data_source.skb.apply_func(load_orders, query)
order_ids = orders[["ID"]].skb.mark_as_X()
fraud_flags = orders["fraud_flag"].skb.mark_as_y()


def load_items(orders_ids, data_source):
    items = data_source["items"]
    return items[items["basket_ID"].isin(order_ids["ID"])]


vectorized_items = order_ids.skb.apply_func(load_items, data_source).skb.apply(
    skrub.TableVectorizer(), exclude_cols="basket_ID"
)


def join_items(orders, vectorized_items):
    return orders.merge(
        vectorized_items.groupby("basket_ID").agg("mean").reset_index(),
        left_on="ID",
        right_on="basket_ID",
    ).drop(columns=["ID", "basket_ID"])


features = order_ids.skb.apply_func(join_items, vectorized_items)
pred = features.skb.apply(HistGradientBoostingClassifier(), y=fraud_flags)

pred.skb.full_report()

(Path(__file__).resolve().parents[1] / "images" / "credit_fraud_graph.svg").write_bytes(
    pred.skb.draw_graph().svg
)
