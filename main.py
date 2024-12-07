import pandas as pd
import json
import ast
from datetime import datetime
import pickle
from tqdm import tqdm
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType, BasicInvertedIndex
from custom_ranker import Ranker, BM25, TF_IDF
import relevance

from llama_tokenise import setup_pipeline


class IR_Ret_Sys:
    def __init__(self):
        
        with open('docid_authors_map.pkl', 'rb') as f:
            self.docid_authors_map = pickle.load(f)
        with open('all_authors.pkl', 'rb') as f:
            self.all_authors = pickle.load(f)
        with open('docid_year_map.pkl', 'rb') as f:
            self.docid_year_map = pickle.load(f)

        with open('docid_title_map.pkl', 'rb') as f:
            self.docid_title_map = pickle.load(f)
        with open('docid_link_map.pkl', 'rb') as f:
            self.docid_link_map = pickle.load(f)
        with open('docid_abstract_map.pkl', 'rb') as f:
            self.docid_abstract_map = pickle.load(f)
        
        with open("stopwords.txt", "r") as file:
            self.stopwords = [line.strip() for line in file]
        
        self.preprocessor = RegexTokenizer()
        
        print('Loading Index')
        self.index = BasicInvertedIndex()
        self.index.load('all_papers_index')
        
        print('Loading Title Index')
        self.title_index = BasicInvertedIndex()
        self.title_index.load('all_papers_title_index')
        
        self.BMRanker = Ranker(self.index, self.title_index, self.preprocessor, stopwords=self.stopwords, scorer=BM25(self.index), 
                          all_authors=list(self.all_authors), docid_to_authors=self.docid_authors_map, docid_to_year=self.docid_year_map,
                               docid_to_abstract = self.docid_abstract_map)
        
        self.TFIDFRanker = Ranker(self.index, self.title_index, self.preprocessor, stopwords=self.stopwords, scorer=TF_IDF(self.index),
                            all_authors=list(self.all_authors), docid_to_authors=self.docid_authors_map, docid_to_year=self.docid_year_map, 
                                  docid_to_abstract = self.docid_abstract_map)

        
        print('Setting up Llama Pipeline.')
        self.text_gen_pipeline = setup_pipeline()
        print('Initialization Complete.')
        
