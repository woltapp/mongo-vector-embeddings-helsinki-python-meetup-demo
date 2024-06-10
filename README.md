# Vector Embeddings Demo
A starter kit for embeddings creation and vector search with Python and MongoDB.

## Getting started

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.11+
  * [MongoDB Atlas](https://www.mongodb.com/products/platform/atlas-database)
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

For embeddings creation and vector search you will also need the following env variables:
* `OPENAI_API_KEY`:
  * Contains an API key for the [OpenAI API](https://platform.openai.com/docs/overview)
* `MONGO_DEMO_URI`:
  * A MongoDB connection string in the following format:\
  `mongodb+srv://username:password@cluster0.example.mongodb.net/database?retryWrites=true&w=majority`\
  (you might also need to add `&tls=true&tlsAllowInvalidCertificates=true`)

## Getting and storing embeddings

Use the command `make create_and_store_embeddings items_path=<path_to_file>` to create and store embeddings for the demo data that is provided. Use `items/demo_items.csv` as `items_path` if you want to compute the embeddings from the provided dataset. Note that this option takes several hours to run, and you will be charged according to OpenAI's [pricing](https://openai.com/api/pricing/). There is also a smaller dataset with precomputed embeddings. Use `items/demo_items_with_embeddings.csv` if you only want to store those to your database.

## Creating indexes
You can find the commands for creating the necessary indexes in the `mongo.ipynb` file.

## Trying it out
`demo.ipynb` contains several commands for performing and visualizing vector search. Note that using the `get_text_embedding` function uses the OpenAI API and is not free.
