from sklearn.cluster import KMeans

def kmeans_embeddings(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(embeddings.numpy())
    return labels