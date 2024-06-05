from io import BytesIO
from typing import Final, Callable, ParamSpec, List

import openai
import pandas as pd
import requests
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import json

from src.repository import EmbeddingsRepository, MongoEmbeddingItem

client = openai.OpenAI()
repository = EmbeddingsRepository()

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model = nn.Sequential(*list(model.children())[:-1])

_TEXT_EMBEDDING_MODEL: Final = "text-embedding-ada-002"

P = ParamSpec("P")


def _error_handler(func: Callable[P, List[float]]) -> Callable[P, List[float]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> List[float]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return []

    return wrapper


@_error_handler
def get_text_embedding(text: str) -> list[float]:
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=_TEXT_EMBEDDING_MODEL)
        .data[0]
        .embedding
    )


@_error_handler
def get_image_embedding(img_url: str) -> list[float]:
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ],
    )
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        embeddings = model(img_tensor)
        embeddings = embeddings.view(embeddings.size(0), -1)

    embedding: list[float] = embeddings.numpy().flatten().tolist()
    return embedding


def create_and_store_embeddings(*, items_path: str) -> None:
    items_df = pd.read_csv(items_path)
    if items_df.empty:
        return

    for _, row in items_df.iterrows():
        item = _create_embedding_item(row)
        if item is not None:
            repository.store_embedding(item)


def _create_embedding_item(row: pd.Series) -> MongoEmbeddingItem | None:
    try:
        return MongoEmbeddingItem(
            venue_name=row["VENUE_NAME"],
            venue_city=row["VENUE_CITY"],
            item_id=row["ITEM_ID"],
            item_name=row["ITEM_NAME"],
            item_image_url=row["ITEM_IMAGE"],
            item_description=row["ITEM_DESCRIPTION"],
            text_embedding=(
                json.loads(row["TEXT_EMBEDDING"])
                if "TEXT_EMBEDDING" in row
                else get_text_embedding(
                    f"""
                        Item Name: {row["ITEM_NAME"]}
                        Item Description: {row["ITEM_DESCRIPTION"][:200]}
                    """,
                )
            ),
            image_embedding=(
                json.loads(row["IMAGE_EMBEDDING"])
                if "IMAGE_EMBEDDING" in row
                else get_image_embedding(row["ITEM_IMAGE"])
            ),
        )
    except Exception as e:
        print(f"Error processing item with ID {row['ITEM_ID']}: {e}")
        return None
