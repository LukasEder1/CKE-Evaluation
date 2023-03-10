import pandas as pd
from nltk.stem import *

def get_stemmed_keywords(df):
    
    """Extract top_k stemmed Keywords from all Deltas in the DataFrame
    
    Args:
        df (DataFrame) : DataFrame with coloumns delta, keyword and score
        top_k (int): Number of Keywords to select per delta
        
    Returns:
        Dictonary of delta-wise stemmed Keywords
    """
    #df.assign(keyword=df["keyword"].astype(str))
    
    
    
    
    
    stemmer = PorterStemmer()
    
    stemmed_kws = list(df["keyword"].astype(str).apply(stemmer.stem))

        
    return stemmed_kws


def number_of_overlaps(set1, set2):
    return len(set1.intersection(set2))


def intersection_over_union(list1, list2):
    """ Computes the IuO score for two lists
    """
    
    set1 = set(list1)
    
    set2 = set(list2)

    area_of_overlap = number_of_overlaps(set1, set2)

    area_of_union = len(set1.union(set2))

    return  area_of_overlap / max(1, area_of_union)


def summary(df_hat, df_baseline,show_metrics=False, k=15):
    
    """ Computes the exact match between the predicted Keywords and the
        Keywords from a Baseline. Furthermore computes the Precision-, Recall-, F1- and IoU-score.
    
    Args:
        df_hat (DataFrame) : Predicted DataFrame with coloumns delta, keyword and score
        df_baseline (DataFrame) : Baseline DataFrame with coloumns delta, keyword and score
        top_k (int): Number of Keywords to select per delta
        
    Returns:
        1. Dictonary of delta-wise matched stemmed Keywords
        2. Dictonary of delta-wise F1-scores
        3. Dictonary of delta-wise Precision-scores
        4. Dictonary of delta-wise Recall-scores
        5. Dictonary of delta-wise IoU-scores
    """
    
    # extract all stemmed keywords for all deltas

    
    stemmed_kws_hat = get_stemmed_keywords(df_hat.head(k))
        
    stemmed_kws_baseline = get_stemmed_keywords(df_baseline.head(k)) 
    

    

    # Perform exact match
    match = list(set([kw_hat for kw_hat in stemmed_kws_hat if kw_hat in stemmed_kws_baseline]))
    
    
    recall = len(match) / max(1, len(stemmed_kws_baseline))
    
    precision = len(match) / max(1, len(stemmed_kws_hat))
    
    # Compute the current F1-Score

    f1 = 2 * (precision * recall) / max(1, precision + recall)
    
    # Compute the current IoU score
    IoU = intersection_over_union(stemmed_kws_hat, stemmed_kws_baseline)
    
    overlaps = number_of_overlaps(set(stemmed_kws_hat), set(stemmed_kws_baseline))
    
    if show_metrics:

        print(f"base: {precision}")

        print(f"hat: {recall}")

        print(f"F1 : {f1}")

        print(f"IoU : {IoU} \n")
        
        print(f"overlaps: {overlaps}")
        
    return f1, precision, recall, IoU, overlaps


def extensive_summary(used_sites, show_results=False, k = 20, use_top_k = True,
                      name_a = "inter_keywords", name_b = "baseline_keywords", save_prefix="", path="dataframes"):
    
    """ Creates a DataFrame for the prediced versus the baseline keywords.
    
    The DataFrame will contain the following columns:
    Index(['Site', 'Delta', 'F1', 'Precision', 'Recall', 'IoU'], dtype='object')
        
    
    Args:
        used_sites (list) : list of sited_ids that should be considered
        show_results (bool) : determines if the DataFrame should be displayed at the end
        
    """
    
    site_id_coloumn = used_sites
    f1s = []
    precs = []
    recalls = []
    IoUs = []
    overlaps = []
        
    for site_id in used_sites:
        
        df_inter = pd.read_csv(f"{path}/{name_a}_{site_id}.csv")

        df_baseline = pd.read_csv(f"{path}/{name_b}_{site_id}.csv")
        
        if use_top_k:
            df_inter = df_inter.head(k)
            
        f1, prec, recall, IoU, overlap = summary(df_inter, df_baseline,show_metrics=False, k=k)
        
        f1s.append(f1)      
        precs.append(prec)
        recalls.append(recall)
        IoUs.append(IoU)
        overlaps.append(overlap)
        
    summary_frame = pd.DataFrame({'Site': site_id_coloumn,
                                  'F1': f1s, 'Precision': precs,
                                  'Recall': recalls, 'IoU': IoUs, '#overlaps': overlaps})
    
    if show_results:
        display(summary_frame.head(10))
    
    summary_frame.to_csv(f"summaries/{save_prefix}summary_furthest_{used_sites[0]}_{used_sites[-1]}.csv", index=False)
    
    
    
