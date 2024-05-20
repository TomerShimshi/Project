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

alpaca_prompt = """you are a jewish Rav, please answer the following question according to the Halakha (Jewish law) .


### Question:
{}

### Answer:
{}"""
def formatting_prompts_func(examples):
    #instruction = "you are a jewish Rav, please answer the following question"
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    #global EOS_TOKEN
    for  input, output in zip( inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        #text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        text = alpaca_prompt.format( input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


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
        base_model_name = "results_fine_tune_after_shulhan_aruch_no_heb_V3\llama-2" #"results_Sulhan_aruch_test\llama-2" #"NousResearch/Llama-2-7b-chat-hf" #
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
    train_dataset_name = "cleaned_Rebe_Q_and_A_dataset_just_rebe_questions_english_no_hebrew_v2.csv"
    train_dataset = load_dataset("csv", data_files=train_dataset_name,split='train[:70%]')#, split="train")
    test_dataset = load_dataset("csv", data_files=train_dataset_name,split='train[-30%:-15%]')
    global EOS_TOKEN
    EOS_TOKEN=tokenizer.eos_token
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True, )
    test_dataset = test_dataset.map(formatting_prompts_func, batched = True,)
    #########################################
    ### Load LoRA Configurations for PEFT ###
    #########################################
    peft_config = LoraConfig(
        lora_alpha = 32,#16,
        lora_dropout=0,#0.1,
        r=8,#64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ##############################
    ### Set Training Arguments ###
    ##############################
    new_model =args.model_name #"tuned-llama-3-8b_V2"
    save_path = os.path.join(os.getcwd() , "results_fine_tune_after_shulhan_aruch_no_heb_V3",new_model)
    temp_save_path = os.path.join(os.getcwd(), "tuning_results")
    print(f"temp save model path = {temp_save_path}")
    training_arguments = TrainingArguments(
        output_dir=temp_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,#4,
        gradient_accumulation_steps=8,#1,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        save_steps=25,
        logging_steps=25,
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
        eval_steps=25,
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
        model=model,
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
    question = "Can I eat pork?"
        
    
    pipe = pipeline(
      task="text-generation",
      model=model,
      tokenizer=tokenizer,
      #eos_token_id=EOS_TOKEN,
      repetition_penalty = 2.0,
      do_sample = True,
      max_new_tokens = 200,
      top_k=10,
      num_return_sequences=1,
    )
    result = pipe( alpaca_prompt.format( question, ""))
    print(result[0]['generated_text'].split(tokenizer.eos_token[0]))

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)