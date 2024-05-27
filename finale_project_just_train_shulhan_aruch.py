import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    Trainer,
    PretrainedConfig,
    DataCollatorForLanguageModeling,
    TextDataset
)
from peft import LoraConfig
from trl import SFTTrainer
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--model_name', type=str,default='llama-2', choices=['llama-2', 'llama-3'],help='chose model name either llama-2 or llama-3')

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Question:
{}

### Answer:
{}"""

def formatting_prompts_func(examples):
    instruction = "you are a jewish Rav, please answer the following question"
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    #global EOS_TOKEN
    for  input, output in zip( inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) #+ EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

def train(args):
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        #bnb_4bit_use_double_quant=False,
    )
    #######################
    ### Load Base Model ###
    #######################
    generate_config = PretrainedConfig(repetition_penalty = 2.0,
      do_sample = True,
      #max_new_tokens = 400,
      #early_stopping = True)
    )
    if args.model_name == 'llama-3':
        base_model_name = "NousResearch/Meta-Llama-3-8B-Instruct" #"unsloth/llama-3-8b-bnb-4bit",#"/content/drive/MyDrive/NLP_proj/results_v3/llama-3" #
    else:
        base_model_name = "NousResearch/Llama-2-7b-chat-hf" #
        model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
         )#{"": 0})

    model.generate_config = generate_config
    print(f"using model = {base_model_name}\n\n")
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
    try: 
        'google.colab' in str(get_ipython())
        base_dir = '/content/drive/MyDrive/NLP_proj'
    except :
        base_dir = os.getcwd()
    train_dataset_name = os.path.join(base_dir, "sorted_kitzur_shulhan_aruch.csv")
    train_dataset = load_dataset("csv", data_files=train_dataset_name,split= "train")
    #test_dataset = load_dataset("csv", data_files=train_dataset_name,split='train[-10%:]')
    print(f"example of the created text : {train_dataset['text'][2]}")
    #########################################
    ### Load LoRA Configurations for PEFT ###
    #########################################
    peft_config = LoraConfig(
        lora_alpha = 32,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ################################
    #Tokenized dataset for trainer
    ################################
    #tokenized_datasets =TextDataset(tokenizer=tokenizer, file_path=file_path,block_size=32)
    #data_collater = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
    ##############################
    ### Set Training Arguments ###
    ##############################
    new_model =args.model_name #"tuned-llama-3-8b_V2"
    save_path = os.path.join(base_dir , "results_Sulhan_aruch",new_model)
    temp_save_path = os.path.join(base_dir, "tuning_results")
    print(f"temp save model path = {temp_save_path}")
    training_arguments = TrainingArguments(
        output_dir=temp_save_path,
        num_train_epochs=2.5,
        per_device_train_batch_size=1,#4,
        gradient_accumulation_steps=8,#1,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        save_steps=6,
        logging_steps=6,
        learning_rate=2e-4,
        warmup_steps= len(train_dataset)//6,
        weight_decay=0.001,
        max_grad_norm=0.3,
        max_steps=-1,
        group_by_length=True,
        lr_scheduler_type= "linear", 
        save_strategy='steps', 
        
    )
    print(f"starting train with args = {training_arguments}")
    ##########################
    ### Set SFT Parameters ###
    ##########################
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    #######################
    ### Fine-Tune Model ###
    #######################
    print("starting model training")
    trainer.train()
    ##################
    ### Save Model ###
    ##################
    print(f"saving model to {save_path}")
    trainer.model.save_pretrained(save_path)
    trainer.tokenizer.save_pretrained(save_path)
    #################
    ### Try Model ###
    #################
    question = "Can I eat pork?"
    pipe = pipeline(
      task="text-generation",
      model=model,
      tokenizer=tokenizer,
      repetition_penalty = 2.0,
      do_sample = True,
      max_new_tokens = 400,
    )
    result =pipe( alpaca_prompt.format(question, ""))
    print(result[0]['generated_text'])

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)