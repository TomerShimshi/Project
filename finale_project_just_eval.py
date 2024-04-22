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

### EVAL ###
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric,HallucinationMetric,ContextualPrecisionMetric
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM

import csv
from tqdm import tqdm

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

        device = "cuda" # the device to load the model onto

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

model_id = "mistralai/Mistral-7B-v0.1"#"mistral-community/Mixtral-8x22B-v0.1"#"mistralai/Mistral-7B-v0.1"#"mistral-community/Mixtral-8x22B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id,
    quantization_config=quant_config,
    device_map="auto"#{"": 0}
)#"mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained(model_id)
#"mistralai/Mistral-7B-v0.1")

mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)
#print(mistral_7b.generate("Write me a joke"))

model_outputs_csv_path = "dataset\\usage_dataset.csv"
eval_csv_dict = csv_to_dict(model_outputs_csv_path)
key = list(eval_csv_dict.keys())[0]
total_iterations = len(eval_csv_dict[key])

for i in tqdm(range(total_iterations), desc="Processing", unit="iterations"):
# Replace this with the actual output from your LLM application
#for i in range(len(eval_csv_dict[list(eval_csv_dict.keys())[0]])):
    
    
    #"We offer a 30-day full refund at no extra cost."

    
    #metric = AnswerRelevancyMetric(
    #    threshold=0.5,
    #    model=mistral_7b,
    #    include_reason=True
    #)
    metric = ContextualPrecisionMetric(threshold=0.5, model=mistral_7b,include_reason=True)
    #we do not have context so we will use the reference output as context
    test_case = LLMTestCase(
        input=eval_csv_dict['question'][i],
        actual_output=eval_csv_dict['actual_output'][i],
        expected_output=eval_csv_dict['expected_output'][i],
        retrieval_context= [eval_csv_dict['expected_output'][i]]
        
    )

    metric.measure(test_case)
    print(metric.score)
    print(metric.reason)

    # or evaluate test cases in bulk
    evaluate([test_case], [metric])

