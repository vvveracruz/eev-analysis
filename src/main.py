import json

from transformers import pipeline

with open("./data/sample.json") as f:
    data = json.load(f)

bertweet_pipe = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
distilbert_pipe = pipeline(model="bhadresh-savani/distilbert-base-uncased-emotion")

tokenized_data = data["content"].split(". ")
with open("./data/sample_tokenized.json", "w") as f:
    json.dump(tokenized_data, f, indent=4)

with open("./data/bertweet_output.json", "w") as f:
    out = bertweet_pipe(tokenized_data)
    json.dump(out, f, indent=4)

with open("./data/distilbert_output.json", "w") as f:
    out = distilbert_pipe(tokenized_data)
    json.dump(out, f, indent=4)
