"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques and build your own tokenizer.
"""


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """

        from nltk import MWETokenizer
        self.lowercase = lowercase

        if multiword_expressions and len(multiword_expressions)>0:
            if lowercase:
                self.mwe_tokenizer = MWETokenizer([tuple(word.lower().split()) for word in multiword_expressions], separator=' ')
            else:
                self.mwe_tokenizer = MWETokenizer([tuple(word.split()) for word in multiword_expressions], separator=' ')
        else:
            self.mwe_tokenizer = None

    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        # TODO: Add support for lower-casing and multi-word expressions

        if self.lowercase:
            input_tokens = [token.lower() for token in input_tokens]
            
        if self.mwe_tokenizer:
            input_tokens = self.mwe_tokenizer.tokenize(input_tokens)
                
        return input_tokens
            

        # if self.lowercase:
        #     input_tokens = [token.lower() for token in input_tokens]

        # if self.multiword_expressions:
        #     processed_tokens = []
        #     i = 0
        #     n = len(input_tokens)

        #     while i < n:
        #         longest_match = input_tokens[i]
        #         match_length = 1

        #         for length in range(min(n - i, max(len(w.split()) for w in self.multiword_expressions)), 1, -1):
        #             candidate = ' '.join(input_tokens[i:i + length])
        #             if candidate in self.multiword_expressions:
        #                 longest_match = candidate
        #                 match_length = length
        #                 break 

        #         processed_tokens.append(longest_match)

        #         i += match_length
            
        #     return processed_tokens
        # else:
        #     return input_tokens



    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # You should implement this in a subclass, not here
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # tokens = text.split()
        # processed_tokens = self.postprocess(tokens)
        # return processed_tokens
        return self.postprocess(text.split())


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        # TODO: Initialize the NLTK's RegexpTokenizer 
        from nltk.tokenize import RegexpTokenizer
        self.token_regex = token_regex
        self.tokenizer = RegexpTokenizer(self.token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        # tokens = self.tokenizer.tokenize(text)
        # return self.postprocess(tokens)
        return self.postprocess(self.tokenizer.tokenize(text))


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        import spacy
        self.tokenizer = spacy.load("en_core_web_sm")
        self.tokenizer.tokenizer.add_special_case("Children's", [{"ORTH": "Children's"}])
        self.tokenizer.tokenizer.add_special_case("Grey's", [{"ORTH": "Grey's"}])
        self.tokenizer.tokenizer.add_special_case("Men's", [{"ORTH": "Men's"}])
        self.tokenizer.tokenizer.add_special_case("Year's", [{"ORTH": "Year's"}])
        self.tokenizer.tokenizer.add_special_case("People's", [{"ORTH": "People's"}])
        self.tokenizer.tokenizer.add_special_case("King's", [{"ORTH": "King's"}])

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single words.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        doc = self.tokenizer(text)
        tokens = [token.text for token in doc]
        return self.postprocess(tokens)
        #pass


# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':

    # import time

    # with open("multi_word_expressions.txt", "r") as file:
    #     multiword_expressions = [line.strip() for line in file]

    # # text = "López Falcón, John- F. KENNEDY, j. R. R. Tolkien, One-Hot Encoding ,  United states census bureau is a famous singer. She lives in New York, Leicester City."
    # # text = 'This is the first document. Apple apple apple.'
    # text = "UNICEF, now officially United Nations Children's Fund, Lupita Nyong'o, Grey's Anatomy, King's College London, International Men's Day, New Year's Eve, People's Liberation Army, Georgia O'Keeffe, is a United Nations agency."

    # # Test SplitTokenizer with multiword expressions
    # split_tokenizer = SplitTokenizer(lowercase=True, multiword_expressions=multiword_expressions)
    # print("SplitTokenizer:", split_tokenizer.tokenize(text))

    # # Test RegexTokenizer with multiword expressions
    # regex_tokenizer = RegexTokenizer(lowercase=True, multiword_expressions=multiword_expressions)
    # print("RegexTokenizer:", regex_tokenizer.tokenize(text))

    # spacy_tokenizer = SpaCyTokenizer(lowercase=True, multiword_expressions=multiword_expressions)
    # print("SpacyTokenizer:", spacy_tokenizer.tokenize(text))

    import json
    import time
    from tqdm import tqdm
    import gzip
    import matplotlib.pyplot as plt
    import os
    
    # Open the .jsonl file
    # with open('wikipedia_200k_dataset.jsonl', 'r') as file:
    #     data = []
    #     for i, document in enumerate(file):
    #         if i >= 1000:
    #             break
    #         data.append(json.loads(document))

    with open('multi_word_expressions.txt', 'r') as file:
        multiword_expressions = [line.strip() for line in file]

    split_tokenizer = SplitTokenizer(lowercase=True, multiword_expressions=multiword_expressions)
    regex_tokenizer = RegexTokenizer(lowercase=True, multiword_expressions=multiword_expressions)
    spacy_tokenizer = SpaCyTokenizer(lowercase=True, multiword_expressions=multiword_expressions)

    tokenizers = {
        'SplitTokenizer': split_tokenizer,
        'RegexTokenizer': regex_tokenizer,
        'SpaCyTokenizer': spacy_tokenizer
    }

    dataset_path = "wikipedia_200k_dataset.jsonl.gz"
    sample_size = 1000

    time_store = {}

    with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
        documents = [json.loads(line)['text'] for _, line in zip(range(sample_size), f)]

    for name, tokenizer in tokenizers.items():
        print(f"Starting tokenization with {name}...")
        start_time = time.time()
        for i, doc in enumerate(documents):
            tokenizer.tokenize(doc)
            if (i + 1) % 100 == 0:
                print(f"{name}: Processed {i + 1}/{sample_size} documents")
        end_time = time.time()
        time_store[name] = end_time - start_time
        print(f"Finished {name} in {time_store[name]:.2f} seconds")

    names = list(time_store.keys())
    times = list(time_store.values())
    
    os.makedirs("output", exist_ok=True)
    plt.bar(names, times)
    plt.xlabel('Tokenizer')
    plt.ylabel('Time Taken (seconds)')
    plt.title('Time Taken for Tokenizing 1000 Documents')
    plt.savefig("output/Q2_plot.png")

    total_doc_size = 200000
    estimated_times = {name: (total_doc_size / sample_size) * time_taken for name, time_taken in time_store.items()}

    with open("Q2_estimated_time.json", 'w') as result_file:
        json.dump(estimated_times, result_file, indent=4)
    
    print("Document Preprocessor run complete")