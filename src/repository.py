import os
from typing import Final

from pydantic import BaseModel, Field
from pymongo import MongoClient

COLLECTION: Final = "embeddings_demo"
MONGO_URI_KEY: Final = "MONGO_DEMO_URI"
TEXT_EMBEDDING_LENGTH: Final = 1536
IMAGE_EMBEDDING_LENGTH: Final = 2048


class MongoEmbeddingItem(BaseModel):
    venue_name: str
    venue_city: str
    item_id: str
    item_name: str
    item_image_url: str
    item_description: str
    text_embedding: list[float] = Field(
        min_length=TEXT_EMBEDDING_LENGTH, max_length=TEXT_EMBEDDING_LENGTH
    )
    image_embedding: list[float] = Field(
        min_length=IMAGE_EMBEDDING_LENGTH, max_length=IMAGE_EMBEDDING_LENGTH
    )


class EmbeddingsRepository:
    def __init__(self) -> None:
        uri = os.environ.get(MONGO_URI_KEY)
        try:
            mongo_client: MongoClient = MongoClient(uri)
            self.db = mongo_client.get_default_database()
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            self.db = None

    def store_embedding(self, item: MongoEmbeddingItem) -> None:
        if self.db is None:
            return
        try:
            self.db[COLLECTION].update_one(
                {"itemId": item.item_id},
                {
                    "$set": {
                        "venueName": item.venue_name,
                        "venueCity": item.venue_city,
                        "itemName": item.item_name,
                        "itemImageUrl": item.item_image_url,
                        "itemDescription": item.item_description,
                        "textEmbedding": item.text_embedding,
                        "imageEmbedding": item.image_embedding,
                    },
                },
                upsert=True,
            )
        except Exception as e:
            print(f"Error storing embedding: {e}")

    def _vector_search(
        self,
        index: str,
        path: str,
        query_vector: list[float],
        limit: int,
        city: str | None,
    ) -> list[dict]:
        if self.db is None:
            return []
        try:
            return list(
                self.db[COLLECTION].aggregate(
                    [
                        {
                            "$vectorSearch": {
                                "index": index,
                                "path": path,
                                "queryVector": query_vector,
                                "numCandidates": limit * 20,
                                "limit": limit,
                                "filter": {"venueCity": city} if city else {},
                            },
                        },
                        {
                            "$project": {
                                "_id": 0,
                                "venueName": 1,
                                "venueCity": 1,
                                "itemName": 1,
                                "itemDescription": 1,
                                "itemImageUrl": 1,
                                "score": {"$meta": "vectorSearchScore"},
                            },
                        },
                    ],
                ),
            )
        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return []

    def search_similar_items(
        self,
        query_vector: list[float],
        limit: int = 10,
        city: str | None = None,
    ) -> list[dict]:
        return self._vector_search(
            "text-embedding-index",
            "textEmbedding",
            query_vector,
            limit,
            city,
        )

    def search_similar_images(
        self,
        query_vector: list[float],
        limit: int = 10,
        city: str | None = None,
    ) -> list[dict]:
        return self._vector_search(
            "image-embedding-index",
            "imageEmbedding",
            query_vector,
            limit,
            city,
        )
