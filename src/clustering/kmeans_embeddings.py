from sklearn.cluster import KMeans

def kmeans_embeddings(embeddings, n_clusters=2):
    """ 
    Performs KMeans clustering on the given embeddings and returns the cluster labels. 
    Please note that this embedding requires the knowledge of the number of clusters (n_clusters) 
    to be specified in advance.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)   #n_init is a parameter of kmeans that means the number of times that kmeans runs with different centroid seeds
    labels = kmeans.fit_predict(embeddings.numpy())
    return labels