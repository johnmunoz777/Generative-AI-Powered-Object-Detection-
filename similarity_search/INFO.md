# Similarity Search for Dataset Integrity
This directory contains the notebooks associated with applying similarity search to our databases.
Specifically, we wanted to verify that we didn't have any data contamination between our testing/validation datasets and our training datasets.
## Techniques used
Image similarity search uses embedding functions to transform images into contextual vectors. These vectors (called embeddings) are stored in a vector store, which allows for querying. The vector store used in our analysis was ChromaDB. The embedding function used is ResNet50, with the classification layer removed.
Both cosine and L2 spaces were used for embeddings. This means when doing similarity comparisons, we looked at the angle of the vectors (smaller angles between vectors means more similar) and the absolute geometric distance between vectors to establish similarity scores.

We were able to find not only exact duplicates between datasets, but also images that had been augmented and cropped. Thus allowing us to quickly find and correct potential data leakage quickly without manually reviewing thousands of images.
