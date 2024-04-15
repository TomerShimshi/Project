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
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
llama_2 = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
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
train_dataset = load_dataset("csv", data_files=train_dataset_name,split='train[:80%]')#, split="train")
test_dataset = load_dataset("csv", data_files=train_dataset_name,split='train[-20%:]')
#########################################
### Load LoRA Configurations for PEFT ###
#########################################
peft_config = LoraConfig(
    lora_alpha = 16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

##############################
### Set Training Arguments ###
##############################
training_arguments = TrainingArguments(
    output_dir="./tuning_results",
    num_train_epochs=2,
    per_device_train_batch_size=12,#4,
    gradient_accumulation_steps=4,#1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)


##########################
### Set SFT Parameters ###
##########################
trainer = SFTTrainer(
    model=llama_2,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

#######################
### Fine-Tune Model ###
#######################
trainer.train()

##################
### Save Model ###
##################
new_model = "tuned-llama-2-7b"
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

#################
### Try Model ###
#################
prompt = "Can I eat pork?"
pipe = pipeline(
  task="text-generation", 
  model=llama_2, 
  tokenizer=tokenizer, 
  max_length=200
)
result = pipe(f"###question \n {prompt}.\n ###answer \n ")
print(result[0]['generated_text'])

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

        device = 'cpu'#"cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"
model_id = "mistral-community/Mixtral-8x22B-v0.1"#"mistralai/Mistral-7B-v0.1"#"mistral-community/Mixtral-8x22B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id)#"mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained(model_id)
#"mistralai/Mistral-7B-v0.1")

mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)
#print(mistral_7b.generate("Write me a joke"))


# Replace this with the actual output from your LLM application
for i in range(len(test_dataset)):
    prompt = test_dataset['quastion'][i]
    pipe = pipeline(
      task="text-generation", 
      model=llama_2, 
      tokenizer=tokenizer, 
      max_length=200
    )
    result = pipe(f"###question \n {prompt}.\n ###answer \n ")
    actual_output = result[0]['generated_text']
    print(actual_output)
    #"We offer a 30-day full refund at no extra cost."

    
    #metric = AnswerRelevancyMetric(
    #    threshold=0.5,
    #    model=mistral_7b,
    #    include_reason=True
    #)
    metric = ContextualPrecisionMetric(threshold=0.5, model=mistral_7b)
    #we do not have context so we will use the reference output as context
    context = [test_dataset['answer'][i]]
    excepected_output = test_dataset['answer'][i]
    test_case = LLMTestCase(
        input=test_dataset['quastion'][i],
        actual_output=actual_output,
        expected_output=test_dataset['answer'][i],
        retrieval_context= context
        
    )

    metric.measure(test_case)
    print(metric.score)
    print(metric.reason)

    # or evaluate test cases in bulk
    evaluate([test_case], [metric])

