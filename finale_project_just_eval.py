import os
import json
import torch
import yaml
import numpy as np
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
from deepeval.evaluate import TestResult
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

import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--dataset_location', type=str,default='dataset\\usage_shulahn_aruch_dataset_RAG.csv', help='location of the csv containing the model outputs for the validation dataset')

def convert_to_serializable(data):
    if isinstance(data, set):
        return list(data)
    return data


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

def test_result_to_dict(test_result: TestResult):
    test_dict ={}
    for metric in test_result.metrics:
        test_dict[metric.__name__] = {}
        
        successful = True
        if metric.error is not None:
            successful = False
        else:
            # This try block is for user defined custom metrics,
            # which might not handle the score == undefined case elegantly
            try:
                if not metric.is_successful():
                    successful = False
            except:
                successful = False

        test_dict[metric.__name__]['is_successful']=successful   
        test_dict[metric.__name__]['score']=metric.score
        test_dict[metric.__name__]['threshold']=metric.threshold
        test_dict[metric.__name__]['strict_mode']=metric.strict_mode
        test_dict[metric.__name__]['evaluation_model']=metric.evaluation_model
        test_dict[metric.__name__]['reason']=metric.reason
        test_dict[metric.__name__]['error']=metric.error
        
        
        
        if metric.score_breakdown:
            for metric_name, score in metric.score_breakdown.items():
                #print(f"      - {metric_name} (score: {score})")
                test_dict[f"{metric_name}_score"] = score #f" {metric_name} (score: {score})"

    #print("")
    #print("For test case:\n")
    
    test_dict['input'] = {test_result.input}
    test_dict["actual output"] = test_result.actual_output
    test_dict["expected output"] =test_result.expected_output
    test_dict["context"] =  test_result.context
    test_dict["retrieval context"] = test_result.retrieval_context
    return test_dict
def save_list_to_json_yaml(input_dict,base_path):
    json_base_path = base_path.replace('yaml', 'json')
    with open(base_path, 'w',encoding='utf-8') as outfile:
            yaml.dump(input_dict, outfile, default_flow_style=False,indent=4)

    with open(json_base_path, 'w',encoding='utf-8') as outfile:
            json.dump(input_dict, outfile,indent=4, default=convert_to_serializable)
    print(f"saved results to {base_path}")

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
 
def eval(args):
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
    save_path =os.path.join( os.getcwd(),'eval_test')
    os.environ["DEEPEVAL_RESULTS_FOLDER"] = save_path
    
    
    

    model_id = "mistralai/Mistral-7B-v0.1" # "yam-peleg/Hebrew-Mistral-7B" # #"mistralai/Mistral-7B-v0.1" #"NousResearch/Meta-Llama-3-8B-Instruct" #"tuned-llama-2-7b"# #"NousResearch/Meta-Llama-3-8B-Instruct" # "unsloth/llama-3-8b-bnb-4bit" #"NousResearch/Meta-Llama-3-8B-Instruct" #"mistralai/Mistral-7B-v0.1"#"mistral-community/Mixtral-8x22B-v0.1"#"mistralai/Mistral-7B-v0.1"#"mistral-community/Mixtral-8x22B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(model_id,
       quantization_config=quant_config,
       device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)
    #print(mistral_7b.generate("Write me a joke"))

    model_outputs_csv_path = args.dataset_location #"dataset\\usage_shulahn_aruch_dataset_RAG.csv" #"dataset\\usage_shulahn_aruch_dataset.csv"
    eval_csv_dict = csv_to_dict(model_outputs_csv_path)
    key = list(eval_csv_dict.keys())[0]
    total_iterations = len(eval_csv_dict[key])
    llm_cases = []
    metric = AnswerRelevancyMetric(threshold=0.5,
                                   #model=mistral_7b,
                                   model = "gpt-3.5-turbo",
                                   include_reason=True,async_mode=True,)
    metric2 =  GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        #evaluation_steps=[
        #"Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        #"You should also heavily penalize omission of detail",
        #"Vague language, or contradicting OPINIONS, are OK"
        #],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        async_mode= True,
        #model = mistral_7b,
        model = "gpt-3.5-turbo"
    )
    for i in tqdm(range(total_iterations), desc="Processing", unit="iterations"):
        print(f"\t running test number {i}  ")
        print("################################")
        test_case = LLMTestCase(
            input=eval_csv_dict['question'][i],
            actual_output=eval_csv_dict['actual_output'][i],
            expected_output=eval_csv_dict['expected_output'][i],
            #retrieval_context= [eval_csv_dict['expected_output'][i+1]]

        )
        llm_cases.append(test_case)

    dataset = EvaluationDataset(test_cases=llm_cases)
    test_results=evaluate(dataset,[metric,metric2],print_results= True,ignore_errors= True, use_cache=True)
    # we need to convert it into a dict and than save it
    #We will convert from test to dict and save to yaml
    all_results =[]
    for test in test_results:
        all_results.append(test_result_to_dict(test))

    save_path_yaml = f"test_eval_save_new_with_eval_steps_{all_results[0]['Answer Relevancy']['evaluation_model']}{'_RAG' if 'RAG' in model_outputs_csv_path else '' }.yaml"
    save_list_to_json_yaml(all_results,save_path_yaml)

    just_successful_answer_relevancy =[result for result in all_results if result['Answer Relevancy']['is_successful'] ]
    just_successful_answer_correctness =[result for result in all_results if  result['Correctness (GEval)']['is_successful']]
    print(f"the accuracy percent we got for answer relevancy is {(len(just_successful_answer_relevancy)/(len(all_results)))*100}%")
    print(f"the accuracy percent we got for answer correctness is {(len(just_successful_answer_correctness)/(len(all_results)))*100}%")
    correctness_scores = [item['Correctness (GEval)']['score'] for item in all_results]
    relevancy_scores = [item['Answer Relevancy']['score'] for item in all_results]
    print(f"the mean score we got for answer correctness is {np.mean(correctness_scores)}")
    print(f"the mean score we got for answer relevancy is {np.mean(relevancy_scores)}")
if __name__ == "__main__":
    args = parser.parse_args()
    eval(args)

