from sentence_comparision import find_additions_deletions_max_ngram
import keyword_extraction
from utilities import *
import numpy as np


def extract_from_tuple_list(baseline_keywords):
    kws = []
    scores = []

    for kw, score in baseline_keywords:
        kws.append(kw)
        scores.append(score)

    return kws, scores

def extract_from_dict(inter_keywords):
    kws = []
    scores = []

    for kw, score in inter_keywords.items():
        kws.append(kw)
        scores.append(score)
    
    return kws, scores

def normalize(keywords):
    
    total_count = sum(keywords.values())
    
    # sort keywords + normalize
    keywords = {k: v/total_count  for k, v in sorted(keywords.items(), key=lambda item: item[1], 
                                                 reverse=True)}
    
    return keywords

############## Baseline 1
def del_as_document(diff_dict):

    
    document = ""
    for position, diff_content in diff_dict.items():
        if len(diff_content) > 0:
            document += " ".join(diff_content) + ". "
            
    return document[:-1]


def add_as_document(diff_dict):

    document = ""
    for position, added_dict in diff_dict.items():
        if len(added_dict) > 0:
            for _,diff_content in added_dict.items():
                if len(diff_content) > 0:
                    document += " ".join(diff_content) + ". "
            
    return document[:-1]


def baseline_diff_content(additions, deletions, ke_extractor, k, max_ngram):
    a = add_as_document(additions)
    
    d = del_as_document(deletions)
    
    
    kws = ke_extractor([a+d], numOfKeywords=k, max_ngram_size=max_ngram)
    
    
    return kws[0]


############## Baseline 2

def flatten_additions(diff_dict):
    document = []
    for position, added_dict in diff_dict.items():
        if len(added_dict) > 0:
            for _,diff_content in added_dict.items():
                if len(diff_content) > 0:
                    document += diff_content
    return document


def flatten_deletions(added_idx):
    document = []
    for position, additions in added_idx.items():
        if len(additions) > 0:
            document += additions
            
    return document


def baseline_keywords_in_diff(documents, ke_extractor, additions, deletions, candidates=50, max_ngram=2):
    
    kw_collection = ke_extractor(documents, numOfKeywords=candidates, max_ngram_size=max_ngram)
    
    former_kws = kw_collection[0]
    later_kws = kw_collection[1]
    
    added_words = flatten_additions(additions)
    
    deleted_words = flatten_deletions(deletions)
    
    baseline_kws = {}
    
    for kw, score in former_kws:
        # check if former keywords are present in the deleted content
        if kw in deleted_words:
            baseline_kws[kw] = baseline_kws.get(kw, 0) + score
    
    for kw, score in later_kws:
        # check if latter keywords are present in the added content
        if kw in added_words:
            baseline_kws[kw] = baseline_kws.get(kw, 0) + score
            
    return normalize(baseline_kws)

### Baslines 3-4 Frequency Based
def baseline3(documents, additions, deletions, maxngram):
    doc_level_stats = build_doc_level_freqs(documents, maxngram=maxngram)
    
    added_words = flatten_additions(additions)
    
    deleted_words = flatten_deletions(deletions)
    
    keywords = {}
    
    for phrase, freq in doc_level_stats[0].items():
        if phrase in deleted_words:
            diff = np.abs(freq - doc_level_stats[1].get(phrase, 0))
            if diff > 0:
                keywords[phrase] = diff
    
    keywords_from_deletions = set(keywords.keys())
    
    missing_added_keywords = set(doc_level_stats[1].keys()) - keywords_from_deletions
    for phrase in missing_added_keywords:
        if phrase in added_words:
            keywords[phrase] = doc_level_stats[1][phrase]
    
    return normalize(keywords)
            

### WITHOUT Using DIFF-Content
def baseline4(documents, maxngram, stopwords):
    doc_level_stats = build_doc_level_freqs(documents, maxngram=maxngram, extra_stopwords=stopwords)
    
    keywords = {}
    
    for phrase, freq in doc_level_stats[0].items():
        diff = np.abs(freq - doc_level_stats[1].get(phrase, 0))
        if diff > 0 and not ngram_in_stopwords(phrase, stopwords):
            keywords[phrase] = diff
    
    keywords_from_deletions = set(keywords.keys())
    
    missing_added_keywords = set(doc_level_stats[1].keys()) - keywords_from_deletions
    for phrase in missing_added_keywords:
        if not ngram_in_stopwords(phrase, stopwords):
            keywords[phrase] = doc_level_stats[1][phrase]
    
    return normalize(keywords)

