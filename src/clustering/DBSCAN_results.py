import pandas as pd
from numpy import ndarray
from sklearn.cluster import DBSCAN

from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics

def dbscan_results(eps, min_samples, my_data, true_labels, macs, output_path="clustering_output/output.csv"):
    """
    This function performs DBSCAN clustering on the provided data and evaluates the results using various metrics.
    It takes the following parameters:  
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: The number of samples (or total weight) in a neighborhood for a
      point to be considered as a core point. This includes the point itself.
    - my_data: The data on which to perform DBSCAN clustering.
    - true_labels: The true labels for the data points.
    """
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(my_data)

    true_labels_filtered = []
    cluster_labels_filtered = []

    for t, c in zip(true_labels, cluster_labels):
        if c != -1:
            true_labels_filtered.append(t)
            cluster_labels_filtered.append(int(c))

    metrics_undiscarded = calc_evaluation_metrics(true_labels_filtered, cluster_labels_filtered)
    print(f"--------------------------------------------------------------")
    print("CALCOLO SENZA RUMORE")
    print(f"ARI: {metrics_undiscarded['ari']:.4f}")            
    print(f"NMI: {metrics_undiscarded['nmi']:.4f}")            
    print(f"Homogeneity: {metrics_undiscarded['homogeneity']:.4f}")            
    print(f"Completeness: {metrics_undiscarded['completeness']:.4f}")          
    print(f"V-measure: {metrics_undiscarded['v_measure']:.4f}")    

    print(f"--------------------------------------------------------------")
    print("Numero di classi:", len(set(true_labels)))
    print(f"Numero di cluster trovati senza rumore: {len(set(cluster_labels_filtered))}")  
    print(f"Cluster labels: {set(cluster_labels_filtered)}")     
    print(f"--------------------------------------------------------------")
  
    if isinstance(my_data, ndarray):
        my_data = pd.DataFrame(my_data)

    output_values = {        
                    "sample_index": my_data.index,          
                    "mac_address": macs,         
                    "true_label": true_labels,            
                    "cluster": cluster_labels,           
                    }
                
    df = pd.DataFrame(output_values)            
    df = df.sort_values("true_label")           
                
    df.to_csv(output_path, index=False)   