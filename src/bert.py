import os

MODEL_PATH = "models/bertopic_model"

# Function to compute embeddings
def compute_embeddings(papers):
    from sentence_transformers import SentenceTransformer
    texts = [paper["title"] for paper in papers]
    model = SentenceTransformer("all-MiniLM-L6-v2")  # You can change this model
    embeddings = model.encode(texts)
    return texts, embeddings

# Function to train/load BERTopic
def compute_topics_with_bertopic(papers, save_model=True):
    
    from bertopic import BERTopic
    texts, embeddings = compute_embeddings(papers)

    # Check if the model exists
    if os.path.exists(MODEL_PATH):
        print("Loading existing BERTopic model...")
        topic_model = BERTopic.load(MODEL_PATH)
    else:
        print("Training new BERTopic model...")

        text_chunks = [texts[i:i+20] for i in range(0, len(texts), 1000)]

        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import IncrementalPCA
        from bertopic.vectorizers import OnlineCountVectorizer

        # Prepare sub-models that support online learning
        umap_model = IncrementalPCA(n_components=5)
        cluster_model = MiniBatchKMeans(n_clusters=10, random_state=0)
        vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=.01)

        # 🔹 Initialize a new BERTopic model with custom settings
        topic_model = BERTopic(umap_model=umap_model,
                       hdbscan_model=cluster_model,
                       vectorizer_model=vectorizer_model)

        # 🔹 Train the model
        for text in text_chunks:
            topic_model.partial_fit(text)

        # 🔹 Save the model after training
        if save_model:
            os.makedirs("models", exist_ok=True)  # Ensure directory exists
            topic_model.save(MODEL_PATH)
            print(f"Model saved at {MODEL_PATH}")

    # Generate topics for the dataset
    topics, _ = topic_model.transform(texts)

    topics = [int(t) for t in topics]

    return topic_model, topics

