create_and_store_embeddings:
	python3 -c "from src.embeddings import create_and_store_embeddings; import sys; \
		create_and_store_embeddings(items_path=sys.argv[1])" $(items_path)
