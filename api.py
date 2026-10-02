import time
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


THRESHOLD = 0.40


df = pd.read_json("datas/df_train.jsonl", orient="records", lines=True)
df_small_train = pd.read_json("datas/df_small_train.jsonl", orient="records", lines=True)
df_small_test = pd.read_json("datas/df_small_test.jsonl", orient="records", lines=True)


model = SentenceTransformer("BAAI/bge-m3")


def prepare_centers(model, dataset):
    intents = sorted(dataset["intent"].unique())

    vectors = model.encode(
        dataset["sentence"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False
    )

    centers = np.stack([
        vectors[dataset["intent"].to_numpy() == intent].mean(axis=0)
        for intent in intents
    ])

    centers /= np.linalg.norm(
        centers,
        axis=1,
        keepdims=True
    )

    return intents, centers


intents, action_vectors = prepare_centers(model, df)

intents_small, centers_small = prepare_centers(
    model,
    df_small_train
)


def predict(query, model):
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    similarities = action_vectors @ query_vector
    top_indices = np.argsort(similarities)[-3:][::-1]

    return [
        {
            "endpoint": intents[index],
            "confidence": float(np.clip(similarities[index], 0, 1)),
        }
        for index in top_indices
    ]


def predict_small(query, model):
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    similarities = centers_small @ query_vector
    top_indices = np.argsort(similarities)[-3:][::-1]

    return [
        {
            "endpoint": intents_small[index],
            "confidence": float(np.clip(similarities[index], 0, 1)),
        }
        for index in top_indices
    ]


app = FastAPI()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


@app.post("/classify")
def classify(request: QueryRequest):
    start_time = time.perf_counter()

    result = predict(request.query, model)

    for item in result:
        item["threshold_passed"] = item["confidence"] >= THRESHOLD
        item["confidence"] = round(item["confidence"], 4)

    print(f"/classify: {time.perf_counter() - start_time:.3f}s")

    return result


@app.post("/prod")
def prod(request: QueryRequest):
    start_time = time.perf_counter()

    result = predict_small(request.query, model)

    for item in result:
        item["threshold_passed"] = item["confidence"] >= THRESHOLD
        item["confidence"] = round(item["confidence"], 4)

    print(f"/prod: {time.perf_counter() - start_time:.3f}s")

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
