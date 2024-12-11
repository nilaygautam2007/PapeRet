
from indexing import InvertedIndex


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, title_index: InvertedIndex, document_preprocessor, stopwords: set[str],
                scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]=None,
                all_authors=[], docid_to_authors={}, docid_to_year={}, docid_to_abstract={}) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.title_index = title_index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict
        self.all_authors = all_authors
        self.docid_to_authors = docid_to_authors
        self.docid_to_year = docid_to_year
        self.docid_to_abstract = docid_to_abstract


    # Random Baseline
    # returns 1000 random (docid, score) tuples    
    def query_random(self, query: str) -> list[tuple[int, float]]:
        import random
        return [(i,10) for i in random.sample(range(92000), 1000)]
        
        
        

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        from tqdm import tqdm
        
        query_tokens = self.tokenize(query)
        
        if self.stopwords:
            query_tokens_wo_stopwords = [token for token in query_tokens if token not in self.stopwords]
        else:
            query_tokens_wo_stopwords = query_tokens
        
        query_word_counts = {}
        for token in query_tokens:
            query_word_counts[token] = query_word_counts.get(token, 0) + 1
        
        possible_docs = set()
        doc_word_counts = {}

        for token in query_tokens_wo_stopwords:
            if token in self.index.index:
                postings = self.index.get_postings(token)
                for doc_id, term_frequency in postings:
                    possible_docs.add(doc_id)
                    if doc_id not in doc_word_counts:
                        doc_word_counts[doc_id] = {}
                    doc_word_counts[doc_id][token] = term_frequency
        
        scored_docs = []
        for doc_id in tqdm(possible_docs): 
            score = self.scorer.score(doc_id, doc_word_counts[doc_id], query_word_counts)
            scored_docs.append((doc_id, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs


    def query_augmented(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        from tqdm import tqdm
        import re
        from collections import defaultdict
        
        query_tokens = self.tokenize(query)
        
        if self.stopwords:
            query_tokens_wo_stopwords = [token for token in query_tokens if token not in self.stopwords]
        else:
            query_tokens_wo_stopwords = query_tokens
        
        query_word_counts = {}
        for token in query_tokens:
            query_word_counts[token] = query_word_counts.get(token, 0) + 1
        
        possible_docs = set()
        doc_word_counts = {}

        # MAIN INDEX - INDEXING
        for token in query_tokens_wo_stopwords:
            if token in self.index.index:
                postings = self.index.get_postings(token)
                if self.index.statistics['index_type'] == 'BasicInvertedIndex':
                    for doc_id, term_frequency in postings:
                        possible_docs.add(doc_id)
                        if doc_id not in doc_word_counts:
                            doc_word_counts[doc_id] = {}
                        doc_word_counts[doc_id][token] = term_frequency
                else: # Positional Index [ Not Required ]
                    for doc_id, term_frequency, _ in postings:
                        possible_docs.add(doc_id)
                        if doc_id not in doc_word_counts:
                            doc_word_counts[doc_id] = {}
                        doc_word_counts[doc_id][token] = term_frequency
        
        scored_docs = []
        for doc_id in tqdm(possible_docs): 
            score = self.scorer.score(doc_id, doc_word_counts[doc_id], query_word_counts)
            scored_docs.append((doc_id, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)


        # TITLE INDEX - INDEXING
        possible_docs_title = set()
        doc_word_counts_title = {}
        for token in query_tokens_wo_stopwords:
            if token in self.title_index.index:
                postings = self.title_index.get_postings(token)
                if self.title_index.statistics['index_type'] == 'BasicInvertedIndex':
                    for doc_id, term_frequency in postings:
                        possible_docs_title.add(doc_id)
                        if doc_id not in doc_word_counts_title:
                            doc_word_counts_title[doc_id] = {}
                        doc_word_counts_title[doc_id][token] = term_frequency
                else: # Positional Index [ Not Required ]
                    for doc_id, term_frequency, _ in postings:
                        possible_docs_title.add(doc_id)
                        if doc_id not in doc_word_counts_title:
                            doc_word_counts_title[doc_id] = {}
                        doc_word_counts_title[doc_id][token] = term_frequency
        
        scored_docs_title = []
        for doc_id in tqdm(possible_docs_title): 
            score = self.scorer.score(doc_id, doc_word_counts_title[doc_id], query_word_counts)
            scored_docs_title.append((doc_id, score))
        
        scored_docs_title.sort(key=lambda x: x[1], reverse=True)

        # COMBINING INDEXING FOR MAIN TEXT AND TITLES
        combined_scores = defaultdict(float)
        
        for docid, score in scored_docs:
            combined_scores[docid] += score*0.15
    
        for docid, score in scored_docs_title:
            combined_scores[docid] += score*0.85
            
        merged_list = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)


        # Step 4: Extract authors mentioned in the query
        authors_in_query = [
            author.lower() for author in self.all_authors
            if re.search(r'\b' + re.escape(author.lower()) + r'\b', query.lower())
        ]
        # print(authors_in_query)
    
        # Step 5: Extract years and detect temporal condition
        temporal_terms = {
            'before': ['before', 'earlier than', 'prior to', 'preceding', 'up to', 'until', 'no later than'],
            'after': ['after', 'later than', 'subsequent to', 'following', 'onward', 'starting from'],
            'between': ['between', 'in the range of', 'from ... to', 'spanning', 'inclusive of', 'in years ... and'],
        }
        
        def detect_temporal_condition(query, temporal_terms):
            condition = None
            for key, keywords in temporal_terms.items():
                if any(keyword in query.lower() for keyword in keywords):
                    condition = key
                    break
            return condition
    
        year_matches = re.findall(r'\b(?:19|20)\d{2}\b', query)
        years_in_query = [int(year) for year in year_matches]
        # print(years_in_query)
    
        temporal_condition = detect_temporal_condition(query, temporal_terms)
        # print(temporal_condition)
    
        relevant_years = []
        if temporal_condition == 'before' and years_in_query:
            relevant_years = [year for year in range(1950, min(years_in_query) + 1)]
        elif temporal_condition == 'after' and years_in_query:
            relevant_years = [year for year in range(max(years_in_query), 2026)]
        elif temporal_condition == 'between' and len(years_in_query) >= 2:
            relevant_years = list(range(min(years_in_query), max(years_in_query) + 1))
        elif years_in_query:
            relevant_years = years_in_query
    
        # Step 6: Filter and rerank top 10000 scored docs
        top_100_docs = merged_list[:10000] # 10000
        
        def rerank_score(doc_id, original_score):
            # Start with the original score
            rerank_score = original_score
            
            # Boost if authors match
            doc_authors = self.docid_to_authors.get(doc_id, [])
            if any(author in doc_authors for author in authors_in_query):
                print('match!')
                rerank_score += 100  # Adjust boost as needed
            
            # Boost if years match
            doc_year = self.docid_to_year.get(doc_id)
            if doc_year in relevant_years:
                rerank_score += 20  # Adjust boost as needed
            
            return rerank_score
        
        reranked_docs = [
            (doc_id, rerank_score(doc_id, score)) for doc_id, score in top_100_docs
        ]
        unique_reranked_docs = {}
        for doc_id, score in reranked_docs:
            unique_reranked_docs[score] = doc_id  # Keeps the last doc_id for each score
        
        # Convert back to a list and sort by score in descending order
        reranked_docs = sorted(unique_reranked_docs.items(), key=lambda x: x[0], reverse=True)
        reranked_docs = [(doc_id, score) for score, doc_id in reranked_docs]  # Convert to original format
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Appending all docs
        remaining_docs = merged_list[10000:]
        reranked_docs.extend(remaining_docs)
        
        return reranked_docs
        

    def query_llama(self, text_gen_pipeline, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # print('LLAMA QUERY')
        from tqdm import tqdm
        from llama_tokenise_rag import setup_pipeline, extract_details
        import ast
        import re
        from collections import defaultdict
        
        # query_tokens = self.tokenize(query)
        query = query.lower()
        text_sample = f'Input Query: "{query}"'
        # text_gen_pipeline = setup_pipeline()

        
        query_dict = extract_details(text_sample, text_gen_pipeline)
        # print('query_dict: \t ', query_dict)
        # query_tokens = query_dict.get('keywords', [])
        # query_tokens_wo_stopwords = query_tokens

        
        query_tokens = self.tokenize(query)
        
        if self.stopwords:
            query_tokens_wo_stopwords = [token for token in query_tokens if token not in self.stopwords]
        else:
            query_tokens_wo_stopwords = query_tokens
        
        query_word_counts = {}
        for token in query_tokens:
            query_word_counts[token] = query_word_counts.get(token, 0) + 1
        
        possible_docs = set()
        doc_word_counts = {}

        # MAIN INDEX - INDEXING
        for token in query_tokens_wo_stopwords:
            if token in self.index.index:
                postings = self.index.get_postings(token)
                if self.index.statistics['index_type'] == 'BasicInvertedIndex':
                    for doc_id, term_frequency in postings:
                        possible_docs.add(doc_id)
                        if doc_id not in doc_word_counts:
                            doc_word_counts[doc_id] = {}
                        doc_word_counts[doc_id][token] = term_frequency
                else: # Positional Index [ Not Required ]
                    for doc_id, term_frequency, _ in postings:
                        possible_docs.add(doc_id)
                        if doc_id not in doc_word_counts:
                            doc_word_counts[doc_id] = {}
                        doc_word_counts[doc_id][token] = term_frequency
        
        scored_docs = []
        for doc_id in tqdm(possible_docs): 
            score = self.scorer.score(doc_id, doc_word_counts[doc_id], query_word_counts)
            scored_docs.append((doc_id, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)


        # TITLE INDEX - INDEXING
        possible_docs_title = set()
        doc_word_counts_title = {}
        for token in query_tokens_wo_stopwords:
            if token in self.title_index.index:
                postings = self.title_index.get_postings(token)
                if self.title_index.statistics['index_type'] == 'BasicInvertedIndex':
                    for doc_id, term_frequency in postings:
                        possible_docs_title.add(doc_id)
                        if doc_id not in doc_word_counts_title:
                            doc_word_counts_title[doc_id] = {}
                        doc_word_counts_title[doc_id][token] = term_frequency
                else: # Positional Index [ Not Required ]
                    for doc_id, term_frequency, _ in postings:
                        possible_docs_title.add(doc_id)
                        if doc_id not in doc_word_counts_title:
                            doc_word_counts_title[doc_id] = {}
                        doc_word_counts_title[doc_id][token] = term_frequency
        
        scored_docs_title = []
        for doc_id in tqdm(possible_docs_title): 
            score = self.scorer.score(doc_id, doc_word_counts_title[doc_id], query_word_counts)
            scored_docs_title.append((doc_id, score))
        
        scored_docs_title.sort(key=lambda x: x[1], reverse=True)

        # COMBINING INDEXING FOR MAIN TEXT AND TITLES
        combined_scores = defaultdict(float)
        
        for docid, score in scored_docs:
            combined_scores[docid] += score*0.15
    
        for docid, score in scored_docs_title:
            combined_scores[docid] += score*0.85
            
        merged_list = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)


        # Step 4: Extract authors mentioned in the query
        authors_in_query_caps = query_dict.get('author', [])
        authors_in_query = [a.lower() for a in authors_in_query_caps]
        # print('authors_in_query: \t', authors_in_query)
        
        # Step 5: Extract years mentioned in the query
        try:
            years_in_query = query_dict.get('year', '')
            if len(years_in_query)>0:
                if 'until' in years_in_query:
                    relevant_years = [year for year in range(1950, int(years_in_query.replace('until ',''))+2 )]
                elif 'onwards' in years_in_query:
                    relevant_years = [year for year in range(int(years_in_query.replace(' onwards','')), 2026)]
                else:
                    relevant_years = [int(years_in_query)]
            else:
                relevant_years = []
        except:
            relevant_years = []

        # print('relevant_years: \t', relevant_years)
    
    
        # Step 6: Filter and rerank top 10000 scored docs
        top_100_docs = merged_list[:10000] # 10000
        
        def rerank_score(doc_id, original_score):
            # Start with the original score
            rerank_score = original_score
            
            # Boost if authors match
            doc_authors = self.docid_to_authors.get(doc_id, [])
            
            if any(author.lower() in doc_authors for author in authors_in_query):
                print('match!')
                rerank_score += 100  # Adjust boost as needed
            
            # Boost if years match
            doc_year = self.docid_to_year.get(doc_id)
            if doc_year in relevant_years:
                rerank_score += 20  # Adjust boost as needed

            doc_abstract_caps = self.docid_to_abstract.get(doc_id)
            doc_abstract = doc_abstract_caps.lower()
            if any(keyword.lower() in doc_abstract for keyword in query_tokens):
                rerank_score += 40
            
            return rerank_score
        
        reranked_docs = [
            (doc_id, rerank_score(doc_id, score)) for doc_id, score in top_100_docs
        ]
        unique_reranked_docs = {}
        for doc_id, score in reranked_docs:
            unique_reranked_docs[score] = doc_id  # Keeps the last doc_id for each score
        
        # Convert back to a list and sort by score in descending order
        reranked_docs = sorted(unique_reranked_docs.items(), key=lambda x: x[0], reverse=True)
        reranked_docs = [(doc_id, score) for score, doc_id in reranked_docs]  # Convert to original format
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Appending all docs
        remaining_docs = merged_list[10000:]
        reranked_docs.extend(remaining_docs)
        
        return reranked_docs
        


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        import numpy as np
        score = 0.0
        doc_len = self.index.get_doc_metadata(docid)['length']  
        N = self.index.statistics['number_of_documents']
        avdl = self.index.statistics['mean_document_length']

        common_words = set(query_word_counts.keys()) & set(doc_word_counts.keys())
        
        for q_term in common_words:
            df_wi = self.index.statistics['num_docs_for_token'][q_term]
            if df_wi > 0:
                doc_tf = doc_word_counts.get(q_term, 0)
                query_tf = query_word_counts.get(q_term, 0)
                score += (np.log((N - df_wi + 0.5) / (df_wi + 0.5))) * (((self.k1 + 1) * doc_tf) / (self.k1 * ((1 - self.b) + self.b * (doc_len / avdl)) + doc_tf)) * (((self.k3 + 1) * query_tf) / (self.k3 + query_tf))

        return score


class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        import numpy as np
        score = 0.0
        N = len(self.index.document_metadata)
        for q_term in query_word_counts:
            if q_term in self.index.index: 
                df_wi = len(self.index.get_postings(q_term))  
                idf = np.log(N / df_wi) + 1                 
                doc_tf = doc_word_counts.get(q_term, 0) 
                if doc_tf > 0:
                    score += np.log(doc_tf + 1) * idf
        return score


