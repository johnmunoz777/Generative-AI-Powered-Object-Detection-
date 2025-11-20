# Similarity Search for Dataset Integrity
This directory contains the notebooks associated with applying similarity search to our databases.
Specifically, we wanted to verify that we didn't have any data contamination between our testing/validation datasets and our training datasets.
## Techniques used
Image similarity search uses embedding functions to transform images into contextual vectors. These vectors (called embeddings) are stored in a vector store, which allows for querying. The vector store used in our analysis was ChromaDB. The embedding function used is ResNet50, with the classification layer removed.
Both cosine and L2 spaces were used for embeddings. This means when doing similarity comparisons, we looked at the angle of the vectors (smaller angles between vectors means more similar) and the absolute geometric distance between vectors to establish similarity scores.

We were able to find not only exact duplicates between datasets, but also images that had been augmented and cropped. Thus allowing us to quickly find and correct potential data leakage quickly without manually reviewing thousands of images.
## Notebooks
- chromadb_collections.ipynb => Embedding Function and creating the vector databases used later
- search_cosine_datasets.ipynb => Initial analysis using vector angles for similarity search
- search_L2_datasets.ipynb => Final analysis using geometric distance for similarity search
- clean_dataset.ipynb => Copy and renaming the Validation dataset based on results from analysis

## Lessons Learned
I would recommend using the L2 (default) space for similarity search, and sorting query results by smallest distances first. I did this last, after starting with cosine space, since I had used it previously for text embeddings and similarity search, where it works much better.
