

"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""

def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    # pass
    relevant_documents = 0
    total_precision = 0.0
    
    for i, rel in enumerate(search_result_relevances[:cut_off]):
        if rel > 0: 
            relevant_documents += 1
            precision_at_i = relevant_documents / (i + 1) 
            total_precision += precision_at_i
    
    if relevant_documents == 0:
        return 0.0
    return total_precision / cut_off


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    # pass
    import math

    def dcg(relevances):
        return sum(rel if i == 0 else rel / math.log2(i+1) for i, rel in enumerate(relevances[:cut_off]))
    
    dcg_value = dcg(search_result_relevances)
    idcg_value = dcg(ideal_relevance_score_ordering) 
    
    if idcg_value == 0:
        return 0.0
    return dcg_value / idcg_value



def run_relevance_tests(relevance_df, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    import pandas as pd
    from tqdm import tqdm
    from llama_tokenise_rag import setup_pipeline

    print('Setting up Llama Pipeline.')
    text_gen_pipeline = setup_pipeline()
    print('Initialization Complete.')
    
    map_scores = []
    ndcg_scores = []

    queries = list(set(relevance_df['Query']))

    for query in tqdm(queries):
        doc_rel_dict = relevance_df.loc[relevance_df['Query'] == query, 
        ['doc_id', 'Lower limit relevance score']].set_index('doc_id')['Lower limit relevance score'].to_dict()
        
        relevances = list(doc_rel_dict.values())
        ideal_relevances = sorted(relevances, reverse=True)

        search_results = ranker.query_llama(text_gen_pipeline=text_gen_pipeline, query=query) ##############################

        search_result_relevances_map = []
        search_result_relevances_ndcg = []
        for doc_id, _ in search_results:
            search_result_relevances_ndcg.append(doc_rel_dict.get(doc_id, 0))
            search_result_relevances_map.append(1 if doc_rel_dict.get(doc_id, 0) in [4, 5] else 0)

        map_scores.append(map_score(search_result_relevances_map))
        ndcg_scores.append(ndcg_score(search_result_relevances_ndcg, ideal_relevances))

    avg_map = sum(map_scores) / len(map_scores)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    
    return {'map': avg_map, 'ndcg': avg_ndcg, 'map_list': map_scores, 'ndcg_list': ndcg_scores}

        


if __name__ == '__main__':
    import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    from tqdm import tqdm
    import pickle
    from document_preprocessor import RegexTokenizer
    from indexing import Indexer, IndexType, BasicInvertedIndex
    from custom_ranker import Ranker, BM25, TF_IDF

    with open("stopwords.txt", "r") as file:
        stopwords = [line.strip() for line in file]

    with open('docid_authors_map.pkl', 'rb') as f:
        docid_authors_map = pickle.load(f)
    with open('all_authors.pkl', 'rb') as f:
        all_authors = pickle.load(f)
    with open('docid_year_map.pkl', 'rb') as f:
        docid_year_map = pickle.load(f)

    with open('docid_title_map.pkl', 'rb') as f:
        docid_title_map = pickle.load(f)
    with open('docid_link_map.pkl', 'rb') as f:
        docid_link_map = pickle.load(f)
    with open('docid_abstract_map.pkl', 'rb') as f:
        docid_abstract_map = pickle.load(f)
    
    
    preprocessor = RegexTokenizer()
    
    print('Loading Index')
    index = BasicInvertedIndex()
    index.load('all_papers_index')
    
    print('Loading Title Index')
    title_index = BasicInvertedIndex()
    title_index.load('all_papers_title_index')
    
    BMRanker = Ranker(index, title_index, preprocessor, stopwords=stopwords, scorer=BM25(index), 
                      all_authors=list(all_authors), docid_to_authors=docid_authors_map, docid_to_year=docid_year_map,
                           docid_to_abstract = docid_abstract_map)
    
    TFIDFRanker = Ranker(index, title_index, preprocessor, stopwords=stopwords, scorer=TF_IDF(index),
                        all_authors=list(all_authors), docid_to_authors=docid_authors_map, docid_to_year=docid_year_map, 
                              docid_to_abstract = docid_abstract_map)

    preprocessor = RegexTokenizer()

    import json

    rankers = {
        'TF_IDF' : TFIDFRanker,
        'BM25' : BMRanker,
    }

    results = {'ranker': [], 'MAP': [], 'NDCG': []}
    all_map_scores = []
    all_ndcg_scores = []
    ranker_names = []

    df1 = pd.read_csv('Relevance_queries_0_20.csv')
    df2 = pd.read_csv('Relevance_queries_20_40-2.csv')
    df = pd.concat([df1, df2])
    
    
    for ranker_name, ranker_type in tqdm(rankers.items()):
        scores = run_relevance_tests(df, ranker_type)
        
        results['ranker'].append(ranker_name)
        results['MAP'].append(scores['map'])
        results['NDCG'].append(scores['ndcg'])

        all_map_scores.extend(scores['map_list'])
        all_ndcg_scores.extend(scores['ndcg_list'])
        ranker_names.extend([ranker_name] * len(scores['map_list']))
        
        print(f"MAP scores for {ranker_name}: {scores['map_list']}")
        print(f"NDCG scores for {ranker_name}: {scores['ndcg_list']}")
    
    results_df = pd.DataFrame(results)
    
    print("\nAverage Performance of Each Ranker:")
    print(results_df)
    results_df.to_csv("final_results_query_llama.csv", index=False) ###########################
    print("Results data has been saved to 'final_results.csv'.")


    plot_data = {
        'ranker': ranker_names + ranker_names,  
        'metric': ['MAP'] * len(all_map_scores) + ['NDCG'] * len(all_ndcg_scores),  
        'score': all_map_scores + all_ndcg_scores 
    }
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv("plot_data_query_llama.csv", index=False) ##############################
    print("Plot data has been saved to 'plot_data.csv'.")



    