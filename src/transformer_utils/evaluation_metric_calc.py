from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score


def calc_evaluation_metrics(true_labels, cluster_labels):
    
    metrics = {}

    # Calcolo delle metriche di valutazione
    metrics["ari"] = adjusted_rand_score(true_labels, cluster_labels)  
    metrics["nmi"] = normalized_mutual_info_score(true_labels, cluster_labels)  
    metrics["homogeneity"] = homogeneity_score(true_labels, cluster_labels)  
    metrics["completeness"] = completeness_score(true_labels, cluster_labels)
    metrics["v_measure"] = v_measure_score(true_labels, cluster_labels)

    return metrics