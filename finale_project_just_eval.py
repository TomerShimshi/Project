import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
import pytest


### EVAL ###
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric,HallucinationMetric,ContextualPrecisionMetric
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

import subprocess
import csv
from tqdm import tqdm
def save_list_to_json(file_path: str, my_list: list):
    with open(file_path, "w") as json_file:
        json.dump(my_list, json_file)   
    print(f"save the list to {file_path}")    
def csv_to_dict(file_name):
    data_dict = {}
    
    with open(file_name, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        keys = csv_reader.fieldnames
        new_dict = {k:[] for k in keys}
        for row in csv_reader:
            # Assuming the first column in the CSV is the key
            for k in keys:
               new_dict[k].append(row[k]) 
            #key = row.pop(next(iter(row)))
            
    
    return new_dict

### EVAL ###

class Mistral7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" #'cpu' #"cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        #model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"
 

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

#######################
### Load Eval Model ###
#######################
os.environ["DEEPEVAL_RESULTS_FOLDER"] = "dataset"
model_id = "mistralai/Mistral-7B-v0.1" #"yam-peleg/Hebrew-Mistral-7B" #"mistralai/Mistral-7B-v0.1" #"NousResearch/Meta-Llama-3-8B-Instruct" #"tuned-llama-2-7b"# #"NousResearch/Meta-Llama-3-8B-Instruct" # "unsloth/llama-3-8b-bnb-4bit" #"NousResearch/Meta-Llama-3-8B-Instruct" #"mistralai/Mistral-7B-v0.1"#"mistral-community/Mixtral-8x22B-v0.1"#"mistralai/Mistral-7B-v0.1"#"mistral-community/Mixtral-8x22B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id,
   quantization_config=quant_config,
   device_map='auto')#{"": 0})
#"mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained(model_id)
#"mistralai/Mistral-7B-v0.1")

mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)
#print(mistral_7b.generate("Write me a joke"))

model_outputs_csv_path = "tuned_llama-3_usage_dataset.csv"
eval_csv_dict = csv_to_dict(model_outputs_csv_path)
key = list(eval_csv_dict.keys())[0]
total_iterations = 2#len(eval_csv_dict[key])//12
#cmd = "deepeval login --confident-api-key pGKpNnRb9JDf2VwL+UZwokgCXeVPlh9W2Ls/9dNqgDU=" 
#os.system(cmd)
#event_id = subprocess.check_output(cmd, shell=True).rstrip()
llm_cases = []
json_path = "tests_results.json"
metric = AnswerRelevancyMetric(threshold=0.5,model=mistral_7b,include_reason=True,async_mode=False,)
metric2 =  GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    async_mode= False,
    model = mistral_7b,
)
for i in tqdm(range(total_iterations-1), desc="Processing", unit="iterations"):
# Replace this with the actual output from your LLM application
#for i in range(len(eval_csv_dict[list(eval_csv_dict.keys())[0]])):
    
    
    #"We offer a 30-day full refund at no extra cost."

    
    
    #we do not have context so we will use the reference output as context
    #print(f"fot the following test we have input : {eval_csv_dict['question'][i]}\n actual output \
    #      = {eval_csv_dict['actual_output'][i]} \n \
    #      expected_output={eval_csv_dict['expected_output'][i]}")
    print(f"\t running test number {i}  ")
    print("################################")
    test_case = LLMTestCase(
        input=eval_csv_dict['question'][i+1],
        actual_output=eval_csv_dict['actual_output'][i+1],
        expected_output=eval_csv_dict['expected_output'][i+1],
        retrieval_context= [eval_csv_dict['expected_output'][i+1]]
        
    )
    llm_cases.append(test_case)
    
    #metric.measure(test_case)
    #print(metric.score)
    #print(metric.reason)

    # or evaluate test cases in bulk
    #temp = evaluate([test_case], [metric,metric2],ignore_errors= True, print_results= True,use_cache=True)
    t=1
    #results.append(temp)
dataset = EvaluationDataset(test_cases=llm_cases)
evaluate(dataset,[metric,metric2],ignore_errors= True, use_cache=True)

