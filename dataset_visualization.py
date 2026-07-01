import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "dataset/train.txt",
    sep=";",
    names=["text", "label"]
)

plt.figure(figsize=(8,5))

df["label"].value_counts().plot(
    kind="bar",
    color="skyblue"
)

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Total Data")

plt.tight_layout()
plt.show()