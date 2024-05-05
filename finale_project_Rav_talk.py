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

base_model_name = os.path.join(os.getcwd() ,"results\\tuned-llama-2-7b")
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
test_dataset = load_dataset("csv", data_files=train_dataset_name,split='train')#[-20%:]')

##############################
### Set Saving Arguments ###
##############################
save_path_csv_path = os.path.join(os.getcwd() , "dataset","usage_dataset.csv")
print(f"temp save model path = {save_path_csv_path}")


# Replace this with the actual output from your LLM application
#for i in range(len(test_dataset)):
for item in tqdm(test_dataset, desc="Processing", unit="items"):
    temp = item['question']#test_dataset['quastion'][i]lh
    prompt =input("Enter question for rav:") #item['question']#test_dataset['quastion'][i]
    pipe = pipeline(
      task="text-generation", 
      model=llama_2, 
      tokenizer=tokenizer, 
      max_length=200
    )
    model_prompt = f"###question \n {prompt}.\n ###answer \n "
    result = pipe(model_prompt)
    actual_output = result[0]['generated_text']
    print(f'the rav answer is: {actual_output}')
    save_dict = {'question': prompt,'actual_output':actual_output, "expected_output":item['answer']}#test_dataset['answer'][i]}
    append_dict_to_csv(save_dict, save_path_csv_path)
    #"We offer a 30-day full refund at no extra cost."

    
    
