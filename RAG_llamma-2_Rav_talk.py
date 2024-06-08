from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from time import time
#import chromadb
#from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from langchain import PromptTemplate


##Just for Eval
import os
from datasets import load_dataset
from tqdm import tqdm
from Rebe_QA_data_scrape_english_site import append_dict_to_csv


compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        #bnb_4bit_use_double_quant=False,
    )

base_model_name = "NousResearch/Llama-2-7b-chat-hf" #
model = AutoModelForCausalLM.from_pretrained(
base_model_name,
quantization_config=quant_config,
device_map="auto")#{"": 0})

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)

# Now we make the querrt pipline 
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        repetition_penalty = 2.0,
        max_new_tokens = 400,
        #top_k=10,
        #num_return_sequences=1,
        )

alpaca_prompt = """you are a jewish Rav, please answer the following question according to the Halakha (Jewish law) .


    ### Question:
    {}

    ### Answer:
    {}"""

def test_model(tokenizer, pipeline, question):
    """
    Perform a query
    print the result
    Args:
        tokenizer: the tokenizer
        pipeline: the pipeline
        prompt_to_test: the prompt
    Returns
        None
    """
    # adapted from https://huggingface.co/blog/llama2#using-transformers
    
    sequences = pipeline(
        alpaca_prompt.format( question, ""),
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens = 400,)
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    
test_model(tokenizer,
           query_pipeline,
           "Can I eat pig?")

llm = HuggingFacePipeline(pipeline=query_pipeline)
# checking again that everything is working fine
llm(prompt="Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words.")

loader = TextLoader("sorted_kitzur_shulhan_aruch.txt",
                    encoding="utf8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

## create embeding using sentsne transformer 
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

##Initialize ChromaDB with the document splits, the embeddings defined previously and with the option to persist it locally.
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

#Initialize chain
retriever = vectordb.as_retriever()

################################
################################
# Test new prompt
################################
################################
template = '''
If you don't know the answer, just say that you don't know.
Don't try to make up an answer.
{context}

Respond as a jewish Rav and please answer the following question according to the Halakha (Jewish law) . %s

Question: {question}
Answer:
'''
prompt = PromptTemplate(
    template=template , 
    input_variables=[
        'context', 
        'question',
    ]
)


qa_new = RetrievalQA.from_chain_type(
    llm=llm, 
    #chain_type="stuff", 
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    #verbose=True
)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    #chain_type_kwargs={"prompt": prompt},
    #verbose=True
)

def test_rag(qa, query,test_pipeline = 'new'):
    #print(f"Query: {query}\n")
    time_1 = time()
    #result = qa.run(query)
    if test_pipeline == 'new':
        result = qa_new.run({"query": query})
    else:
       result = qa.run(query) 
    time_2 = time()
    print(f"Inference time: {round(time_2-time_1, 3)} sec.")
    #print("\nResult: ", result)
    return result

query = "Can I work on shabbes?"
prompet = alpaca_prompt.format( query, "")
test_rag(qa, prompet)


################################################
#   NOW we want to test the validation csv     #
################################################





# Replace this with the actual output from your LLM application
#for i in range(len(test_dataset)):
question = input('Please enter a question for the Rav \n Enter empty string to quit \n')
while len(question)>1:   
    
    model_prompt_custom_prompt = question#alpaca_prompt.format( question, "")
    model_prompt = alpaca_prompt.format( question, "")
    
    result_custom_prompt = test_rag(qa,query=model_prompt_custom_prompt,test_pipeline='new')
    actual_output_custom_prompt = result_custom_prompt.split("Answer:")[1]
    
    result_default_prompt = test_rag(qa,query=model_prompt,test_pipeline='old')
    actual_output = result_default_prompt.split("Helpful Answer:")[1]
    print(f"The Rav answer with custom prompt is {actual_output_custom_prompt} \n \n")
    
    print(f"The Rav answer with default prompt is {actual_output} \n \n")
    question = input('Please enter a question for the Rav \n Enter empty string to quit \n')
