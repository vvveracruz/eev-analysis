import json

import pandas as pd
import plotly.express as px

with open("./data/bertweet_output.json") as f:
    data = json.load(f)

with open("./data/sample_tokenized.json") as f:
    tokenized_data = json.load(f)

df = pd.DataFrame(data=data)

df["label_value"] = df["label"].replace({"NEG": -1, "POS": 1, "NEU": 0})
df["counter"] = abs(df["label_value"]).cumsum()
df["data"] = tokenized_data

df["sentiment_confidence"] = df["score"] * df["label_value"]

df["pure_sentiment_cumsum_normalised"] = df["label_value"].cumsum() / df["counter"]
# this one is kind of bs bc the sentiment value and the confidence value are different things
# it doesn't really make sense to put them together
df["sentiment_confidence_cumsum_normalised"] = (
    df["sentiment_confidence"].cumsum() / df["counter"]
)

print(df)

fig = px.line(
    df,
    y=["pure_sentiment_cumsum_normalised"],
    hover_data="data",
    title="Sentiment over time",
)
fig.show()
