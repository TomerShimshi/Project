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
        max_new_tokens = 100,)

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
    alpaca_prompt = """you are a jewish Rav, please answer the following question according to the Halakha (Jewish law) .


    ### Question:
    {}

    ### Answer:
    {}"""
    sequences = pipeline(
        alpaca_prompt.format( question, ""),
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens = 100,)
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

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
def test_rag(qa, query):
    print(f"Query: {query}\n")
    time_1 = time()
    result = qa.run(query)
    time_2 = time()
    print(f"Inference time: {round(time_2-time_1, 3)} sec.")
    print("\nResult: ", result)

query = "Can I work on shabbes?"
test_rag(qa, query)


###########################
#NOW we want to test the validation csv