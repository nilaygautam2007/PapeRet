'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
from document_preprocessor import Tokenizer
from collections import Counter, defaultdict

class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter() # token count
        self.vocabulary = set()  # the vocabulary of the collection
        self.document_metadata = {} # metadata like length, number of unique tokens of the documents

        self.index = defaultdict(list)  # the index 

    
    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        #raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        #raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        #raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        #raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        #raise NotImplementedError
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        #raise NotImplementedError

    def save(self) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        #raise NotImplementedError

    def load(self) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        #raise NotImplementedError





########################################

class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.statistics['num_docs_for_token'] = defaultdict(int)
        self.statistics['num_occs_for_token'] = defaultdict(int)
        self.statistics['unique_token_count'] = 0
        self.statistics['total_token_count'] = 0
        self.statistics['number_of_documents'] = 0
        self.statistics['mean_document_length'] = 0


    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        if docid in self.document_metadata:
            del self.document_metadata[docid]

        for token in list(self.index.keys()):
            self.index[token] = [(doc, freq) for doc, freq in self.index[token] if doc != docid]
            if not self.index[token]:
                del self.index[token]
                self.vocabulary.remove(token)

        extra_stats = self.get_statistics()

        self.statistics['unique_token_count'] = extra_stats['unique_token_count']
        self.statistics['total_token_count'] = extra_stats['total_token_count']
        self.statistics['number_of_documents'] = extra_stats['number_of_documents']
        self.statistics['mean_document_length'] = extra_stats['mean_document_length']

        for token in self.index.keys():
            posting = self.get_postings(token)
            self.statistics['num_docs_for_token'][token] = len(posting)
            self.statistics['num_occs_for_token'][token] = sum([doc[1] for doc in posting])

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        token_counts = Counter(tokens)
        self.document_metadata[docid] = {'unique_tokens': len(token_counts), "length": len(tokens)}
        for token, count in token_counts.items():
            self.index[token].append((docid, count))
            self.vocabulary.add(token)
            self.statistics['vocab'][token] += count
            self.statistics['num_docs_for_token'][token] += 1
            self.statistics['num_occs_for_token'][token] += count
            self.statistics['total_token_count'] += count
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['number_of_documents'] += 1
        self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents'] if self.statistics['number_of_documents'] > 0 else 0
        

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        return self.index[term]

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata[doc_id]

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        term_count = sum(freq for _, freq in self.index[term])
        doc_frequency = len(self.index[term])
        return {
            "term_count": term_count,
            "doc_frequency": doc_frequency
        }
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        total_token_count = sum(self.statistics['vocab'].values())
        number_of_documents = len(self.document_metadata)
        mean_document_length = total_token_count / number_of_documents if number_of_documents > 0 else 0

        return {
            "unique_token_count": len(self.index),
            "total_token_count": total_token_count,
            "number_of_documents": number_of_documents,
            "mean_document_length": mean_document_length
        }

    def save(self, index_directory_name) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        import json
        import os
        os.makedirs(index_directory_name, exist_ok=True)
        data_to_save = {
            'index': self.index,  
            'statistics': self.statistics,
            'vocabulary': list(self.vocabulary),  
            'document_metadata': self.document_metadata  
        }
        with open(os.path.join(index_directory_name, 'index.json'), 'w') as f:
            json.dump(data_to_save, f)
        print(f'Index saved to {index_directory_name}')

    def load(self, index_directory_name) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        import json
        import os
        with open(os.path.join(index_directory_name, 'index.json'), 'r') as f:
            data = json.load(f)

        self.index = data['index']
        self.statistics = data['statistics']
        self.vocabulary = set(data['vocabulary'])
        self.document_metadata = {int(k): v for k, v in data['document_metadata'].items()}
        
        print(f'Index loaded from {index_directory_name}')



class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        self.statistics['index_type'] = 'PositionalIndex'

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        if docid in self.document_metadata:
            del self.document_metadata[docid]

        for token in list(self.index.keys()):
            self.index[token] = [(doc, freq, pos) for doc, freq, pos in self.index[token] if doc != docid]
            if not self.index[token]:
                del self.index[token]
                self.vocabulary.remove(token)

        extra_stats = self.get_statistics()

        self.statistics['unique_token_count'] = extra_stats['unique_token_count']
        self.statistics['total_token_count'] = extra_stats['total_token_count']
        self.statistics['number_of_documents'] = extra_stats['number_of_documents']
        self.statistics['mean_document_length'] = extra_stats['mean_document_length']

        for token in self.index.keys():
            posting = self.get_postings(token)
            self.statistics['num_docs_for_token'][token] = len(posting)
            self.statistics['num_occs_for_token'][token] = sum([doc[1] for doc in posting])

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        token_positions = defaultdict(list)
        for position, token in enumerate(tokens):
            token_positions[token].append(position)

        self.document_metadata[docid] = {
            "unique_tokens": len(token_positions),
            "length": len(tokens)
        }

        for token, positions in token_positions.items():
            self.index[token].append((docid, len(positions), positions))
            self.vocabulary.add(token)
            self.statistics['vocab'][token] += len(positions)
            self.statistics['num_docs_for_token'][token] += 1
            self.statistics['num_occs_for_token'][token] += len(positions)
            self.statistics['total_token_count'] += len(positions)
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['number_of_documents'] += 1
        self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents'] if self.statistics['number_of_documents'] > 0 else 0

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        term_count = sum(len(pos) for _,_, pos in self.index[term])
        doc_frequency = len(self.index[term])
        return {
            "term_count": term_count,
            "doc_frequency": doc_frequency
        }

    def get_postings(self, term: str) -> list:
        return self.index.get(term, [])

    
        
    

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                        document_preprocessor: Tokenizer, stopwords: set[str],
                        minimum_word_frequency: int, text_key = "text",
                        max_docs: int = -1, ) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.

        Returns:
            An inverted index
        
        '''
        import os, json, gzip
        from tqdm import tqdm

        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            raise ValueError("Unsupported index type")

        # # # JSONL
        # # with open(dataset_path, 'r') as file:
        # #     for line_num, line in enumerate(file):
        # #         if max_docs != -1 and line_num >= max_docs:
        # #             break
        # #         document = json.loads(line.strip())
        # #         doc_id = document.get('docid')  # Get the docid from the JSON object
        # #         tokens = document_preprocessor.tokenize(document[text_key])
        # #         index.add_doc(doc_id, tokens)
                
        # # JSONL.GZ     
        # if max_docs > 0:
        #     with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
        #         documents = [json.loads(line) for _, line in zip(range(max_docs), f)]
        # else:
        #     with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
        #         documents = [json.loads(line) for line in f]
        # for doc in tqdm(documents):
        #     doc_id = doc.get('docid')
        #     tokens = document_preprocessor.tokenize(doc[text_key])
        #     index.add_doc(doc_id, tokens)

        if dataset_path.endswith('.gz'):
            with gzip.open(dataset_path, 'rt', encoding='utf-8') as file:
                for line_num, line in enumerate(file):
                    if max_docs != -1 and line_num >= max_docs:
                        break
                    document = json.loads(line.strip())
                    doc_id = document.get('docid')
                    tokens = document_preprocessor.tokenize(document[text_key])
                    # if doc_augment_dict and len(doc_augment_dict)>0:            ####
                    #     doc_augment_queries = doc_augment_dict.get(doc_id, [])  ####
                    #     for q in doc_augment_queries:                           ####
                    #         tokens.extend(document_preprocessor.tokenize(q))    ####
                    index.add_doc(doc_id, tokens)
        else:
            with open(dataset_path, 'r') as file:
                idx=0
                for line_num, line in enumerate(file):
                    idx+=1
                    if max_docs != -1 and line_num >= max_docs:
                        break
                    document = json.loads(line.strip())
                    
                    # doc_id = document.get('docid') 
                    doc_id = idx
                    tokens = document_preprocessor.tokenize(document[text_key])
                    # if doc_augment_dict and len(doc_augment_dict)>0:            ####
                    #     doc_augment_queries = doc_augment_dict.get(doc_id, [])  ####
                    #     for q in doc_augment_queries:                           ####
                    #         tokens.extend(document_preprocessor.tokenize(q))    ####
                    index.add_doc(doc_id, tokens)
                    if doc_id%10000 == 0:
                        print(f'{doc_id} documents done.')

        
        if stopwords:
            for token in list(index.index.keys()):
                if token in stopwords:
                    del index.index[token]

        if index_type == IndexType.BasicInvertedIndex:
            if minimum_word_frequency > 0:
                for token in list(index.index.keys()):
                    if sum(freq for _, freq in index.index[token]) < minimum_word_frequency:
                        del index.index[token]
        elif index_type == IndexType.PositionalIndex:
            if minimum_word_frequency > 0:
                for token in list(index.index.keys()):
                    if sum(freq for _, freq, _ in index.index[token]) < minimum_word_frequency:
                        del index.index[token]

        extra_stats = index.get_statistics()

        index.statistics['unique_token_count'] = extra_stats['unique_token_count']
        index.statistics['total_token_count'] = extra_stats['total_token_count']
        index.statistics['number_of_documents'] = extra_stats['number_of_documents']
        index.statistics['mean_document_length'] = extra_stats['mean_document_length']

        for token in index.index.keys():
            posting = index.get_postings(token)
            index.statistics['num_docs_for_token'][token] = len(posting)
            index.statistics['num_occs_for_token'][token] = sum([doc[1] for doc in posting])

        return index

# TODO for each inverted index implementation, use the Indexer to create an index with the first 10, 100, 1000, and 10000 documents in the collection (what was just preprocessed). At each size, record (1) how
# long it took to index that many documents and (2) using the get memory footprint function provided, how much memory the index consumes. Record these sizes and timestamps. Make
# a plot for each, showing the number of documents on the x-axis and either time or memory
# on the y-axis.

'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1
    
    def save(self):
        print('Index saved!')


if __name__ == '__main__':

    import time, tracemalloc, os
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    def analyze_efficiency(indexer_class, tokenizer, dataset_path, stopwords, index_directory_name, minimum_word_frequency=0):
        sizes = [10, 100, 1000, 10000]
        times = []
        memories = []
        ram = []

        for size in tqdm(sizes):
            start_time = time.time()
            tracemalloc.start()

            index = indexer_class.create_index(IndexType.PositionalIndex,
                                            dataset_path, tokenizer, stopwords, minimum_word_frequency=0, max_docs=size)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            end_time = time.time()

            index.save(index_directory_name)
            
            index_file_path = os.path.join(index_directory_name, 'index.json')
            file_size_in_mb = os.path.getsize(index_file_path) / (1024 * 1024)  # Convert bytes to MB
            memories.append(file_size_in_mb)
            
            times.append(end_time - start_time)
            ram.append(peak / (1024 * 1024))  # Memory in MB

        return sizes, times, memories, ram


    # Plotting the results
    def plot_efficiency(sizes, times, memories, ram):
        print(sizes)
        print(times)
        print(memories)

        plt.figure(1, figsize = [10.4, 4.8])
        plt.subplot(1,3,1) 
        plt.plot(sizes, times, c='blue', label='Time', marker='o')
        plt.xlabel('Number of Documents')
        plt.ylabel('Time (seconds)', c='blue')
        plt.subplot(1,3,2)
        plt.plot(sizes, memories, c='green', label='Memory', marker='o')
        plt.xlabel('Number of Documents')
        plt.ylabel('Memory (MB)', c='green')
        plt.subplot(1,3,3)
        plt.plot(sizes, ram, c='red', label='RAM', marker='o')
        plt.xlabel('Number of Documents')
        plt.ylabel('RAM (MB)', c='red')
        plt.tight_layout()
        plt.savefig("PositionalIndexMemoriesTimes3.png")
        plt.show()
        # fig, ax1 = plt.subplots()
        
        # ax1.set_xlabel('Number of Documents')
        # ax1.set_ylabel('Time (seconds)', color='tab:blue')
        # ax1.plot(sizes, times, color='tab:blue', label='Time')
        # ax1.tick_params(axis='y', labelcolor='tab:blue')

        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Memory (MB)', color='tab:green')
        # ax2.plot(sizes, memories, color='tab:green', label='Memory')
        # ax2.tick_params(axis='y', labelcolor='tab:green')

        # fig.tight_layout()
        # plt.title('Indexing Time and Memory Usage')
        # plt.show()

    import document_preprocessor as dp

    with open("multi_word_expressions.txt", "r") as file:
        multiword_expressions = [line.strip() for line in file]

    with open("stopwords.txt", "r") as file:
        stopwords = [line.strip() for line in file]

    tokenizer = dp.RegexTokenizer(multiword_expressions=multiword_expressions)
    dataset_path = 'wikipedia_200k_dataset.jsonl'

    sizes, times, memories, ram = analyze_efficiency(Indexer, tokenizer, dataset_path, stopwords=stopwords, index_directory_name = 'indexing_question_temp')
    plot_efficiency(sizes, times, memories, ram)

