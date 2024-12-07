import ast
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

def setup_pipeline():
    login("")
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,
        truncation=True
    )
    text_gen_pipeline.tokenizer.pad_token_id = text_gen_pipeline.model.config.eos_token_id
    set_seed(42)
    return text_gen_pipeline

def extract_details(text_sample, text_gen_pipeline):
    messages = [
        {
            "role": "system",
            "content": '''Your task is to extract the author's name, year (or year range), and identify important keywords from the query. For queries mentioning conditions such as "after 2019" or "before 2019," convert them into appropriate year ranges. For example:

            - "after YYYY" should be interpreted as a year greater than YYYY (e.g., "YYYY+1 onwards").
            - "before 2019" should be interpreted as a year less than 2019 (e.g., "until 2018").

            Output the results strictly in this JSON format: 
            {
                "author": ["<author1>", "<author2>", ...],
                "year": "<year or year range>",
                "keywords": ["<keyword1>", "<keyword2>", ...]
            }.

            For example:
            Input Query: "Show me papers by Geoffrey Hinton on deep learning published before 2019."
            Output: {"author": ["Geoffrey Hinton"], "year": "until 2018", "keywords": ["deep", "learning"]}

            Input Query: "Machine learning for healthcare applications"
            Output: {"author": "", "year": "", "keywords": ["Machine", "learning", "healthcare"]}

            Input Query: "Show papers before 2019 by Yoshua Bengio and Ashish Vaswani."
            Output: {"author": ["Yoshua Bengio", "Ashish Vaswani"], "year": "until 2018", "keywords": []}

            Make sure to capture the year or range accurately based on the query. Extract key terms from the query and include them as a list of keywords. Provide the output in JSON format only.'''
        },
        {"role": "user", "content": text_sample},
    ]

    response = text_gen_pipeline(messages, max_new_tokens=50)
    generated_text = response[0]["generated_text"]
    generated_text = generated_text[-1]['content']
    output_dict = ast.literal_eval(generated_text.strip().replace('\n', ''))
    return output_dict


def summarize_documents_with_query(abstracts, query, generator_pipeline):
    input_text = (
        f"User query: {query}\n\n"
        "Summarize the following " + str(len(abstracts)) +" paper abstracts with respect to the query:\n" +
        "\n\n".join(abstracts) + "\n Please summarise the abstracts to perfectly answer the User query. Only give back the summary to me. Do not return anything else."
    )
    print(len(input_text))
    summary = generator_pipeline(input_text, max_new_tokens=1000, truncation=True)
    summary = summary[0]["generated_text"]
    start_index = summary.find("Do not return anything else.")
    end_index = start_index + len("Do not return anything else.") +3
    summary = summary[end_index:]
    return summary

if __name__ == "__main__":
    text_sample = 'Input Query: "Show me papers by Ashish Vaswani published after 2012"'
    text_gen_pipeline = setup_pipeline()
    output_dict = extract_details(text_sample, text_gen_pipeline)
    print(output_dict)
    top_10_abstracts = [
        "Abstract 1: This paper discusses the use of transformers in NLP...",
        "Abstract 2: The study explores methods for improving machine translation...",
        "Abstract 3: This work introduces novel attention mechanisms...",
    ]
    user_query = "Explain the role of transformers in NLP"

    # Summarize using the query
    summary = summarize_documents_with_query(top_10_abstracts, user_query, text_gen_pipeline)
    
    print("Summary:", summary)
    