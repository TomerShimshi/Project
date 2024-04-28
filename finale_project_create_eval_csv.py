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
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Question:
{}

### Answer:
{}"""
base_model_name = os.path.join(os.getcwd() ,"results\\tuned-llama-3-8b")
llama_2 = AutoModelForCausalLM.from_pretrained(
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
train_dataset_name = "Rebe_Q_and_A_dataset_just_rebe_questions_english.csv"
test_dataset = load_dataset("csv", data_files=train_dataset_name,split='train[-15%:]')

##############################
### Set Saving Arguments ###
##############################
save_path_csv_path = os.path.join(os.getcwd() , "dataset","usage_dataset.csv")
print(f"temp save model path = {save_path_csv_path}")


# Replace this with the actual output from your LLM application
#for i in range(len(test_dataset)):
instruction = "you are a jewish Rav, please answer the following question"
for item in tqdm(test_dataset, desc="Processing", unit="items"):
    question = item['question']#test_dataset['quastion'][i]
    
    pipe = pipeline(
      task="text-generation", 
      model=llama_2, 
      tokenizer=tokenizer, 
      max_length=2000
    )
    model_prompt = alpaca_prompt.format(instruction, question, "")
    
    result = pipe(model_prompt)
    actual_output = result[0]['generated_text'].split("### Answer:")[0]
    save_dict = {'question': question,'actual_output':actual_output, "expected_output":item['answer']}#test_dataset['answer'][i]}
    append_dict_to_csv(save_dict, save_path_csv_path)
    #"We offer a 30-day full refund at no extra cost."

    
    
