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
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--model_name', type=str,default='llama-2', choices=['llama-2', 'llama-3'],help='chose model name either llama-2 or llama-3')


def train(args):
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
    if args.model_name == 'llama-3':
        
        base_model_name = "unsloth/llama-3-8b-bnb-4bit" #"unsloth/llama-3-8b-bnb-4bit"#"NousResearch/Meta-Llama-3-8B" #"NousResearch/Llama-2-7b-chat-hf" #"unsloth/llama-3-8b-bnb-4bit"#"NousResearch/Meta-Llama-3-8B" #"NousResearch/Meta-Llama-3-8B-Instruct" #"NousResearch/Llama-2-7b-chat-hf" # "NousResearch/Meta-Llama-3-8B-Instruct" #
    else:
        base_model_name = "NousResearch/Llama-2-7b-chat-hf" #
    llama_3 = AutoModelForCausalLM.from_pretrained(
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
    train_dataset = load_dataset("csv", data_files=train_dataset_name,split='train[:70%]')#, split="train")
    test_dataset = load_dataset("csv", data_files=train_dataset_name,split='train[-30%:-15%]')
    #########################################
    ### Load LoRA Configurations for PEFT ###
    #########################################
    peft_config = LoraConfig(
        lora_alpha = 32,#16,
        lora_dropout=0.1,
        r=8,#64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ##############################
    ### Set Training Arguments ###
    ##############################
    new_model =args.model_name #"tuned-llama-3-8b_V2"
    save_path = os.path.join(os.getcwd() , "results",new_model)
    temp_save_path = os.path.join(os.getcwd(), "tuning_results")
    print(f"temp save model path = {temp_save_path}")
    training_arguments = TrainingArguments(
        output_dir=temp_save_path,
        num_train_epochs=6,
        per_device_train_batch_size=1,#4,
        gradient_accumulation_steps=8,#1,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        save_steps=50,
        logging_steps=50,
        learning_rate=2e-4,
        warmup_steps= len(train_dataset)//6,
        weight_decay=0.001,
        tf32=False,
        fp16=True,
        #bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        #warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type= "linear", #"constant",
        load_best_model_at_end=True,
        #save_strategy='epoch',
        evaluation_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        eval_accumulation_steps=1,
        per_device_eval_batch_size=1
        #torch_compile=True,
    )
    print(f"starting train with args = {training_arguments}")

    ##########################
    ### Set SFT Parameters ###
    ##########################
    trainer = SFTTrainer(
        model=llama_3,
        train_dataset=train_dataset,
        eval_dataset= test_dataset,
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
    prompt = "Can I eat pork?"
    pipe = pipeline(
      task="text-generation", 
      model=llama_3, 
      tokenizer=tokenizer, 
      max_length=200
    )
    result = pipe(f"###question \n {prompt}.\n ###answer \n ")
    print(result[0]['generated_text'])

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)