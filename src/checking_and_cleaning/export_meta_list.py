import ijson
import numpy as np

ids = []
with open("metadata.json", "rb") as fin:
    for item in ijson.items(fin, "item"):
        ids.append(item["id"])

ids = np.array(ids)
ids = np.unique(ids)
# ids = ids[ids<666_000]
np.save("ids2.npy", ids)