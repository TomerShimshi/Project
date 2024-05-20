import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig
from trl import SFTTrainer
from tqdm import tqdm

from Rebe_QA_data_scrape_english_site import append_dict_to_csv
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

#######################
### Load Base Model ###
#######################

base_model_name = "tomer-shimshi/llama2-Rav" #os.path.join(os.getcwd() ,"results_fine_tune_after_shulhan_aruch_no_heb_V3\llama-2")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto"#{"": 0}
)

######################
### Load Tokenizer ###
######################
tokenizer = AutoTokenizer.from_pretrained(
  base_model_name, 
  trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

####################
### Load Dataset ###
####################
train_dataset_name = "cleaned_Rebe_Q_and_A_dataset_just_rebe_questions_english_no_hebrew.csv"
test_dataset = load_dataset("csv", data_files=train_dataset_name,split='train')#[-20%:]')

##############################
### Set Saving Arguments ###
##############################
save_path_csv_path = os.path.join(os.getcwd() , "dataset","usage_dataset.csv")
alpaca_prompt = """you are a jewish Rav, please answer the following question according to the Halakha (Jewish law) .


### Question:
{}

### Answer:
{}"""
def formatting_prompts_func(examples):
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    #global EOS_TOKEN
    for  input, output in zip( inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        #text = alpaca_prompt.format(instruction, input, output) #+ EOS_TOKEN
        text = alpaca_prompt.format( input, output) #+ EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# Replace this with the actual output from your LLM application
#for i in range(len(test_dataset)):
question = input('Please enter a question for the Rav \n Enter empty string to quite \n')
while len(question)>1:
    ##question = item['question']#test_dataset['quastion'][i]
    
    pipe = pipeline(
      task="text-generation",
      model=model,
      tokenizer=tokenizer,
      #eos_token_id=EOS_TOKEN,
      repetition_penalty = 2.0,
      do_sample = True,
      max_new_tokens = 400,
      #top_k=10,
      #num_return_sequences=1,
      
    )
    model_prompt = alpaca_prompt.format( question, "")
    
    result = pipe(model_prompt)
    actual_output = result[0]['generated_text'].split("### Answer:")[1]
   
    #append_dict_to_csv(save_dict, save_path_csv_path)
    print(f"The Rav answer is {actual_output} \n \n")
    question = input('Please enter a question for the Rav \n Enter empty string to quite \n')#item['question']#test_dataset['quastion'][i]
    #"We offer a 30-day full refund at no extra cost."

    
    
