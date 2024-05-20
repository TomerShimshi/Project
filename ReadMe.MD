# TRAIN
Please do the following in order to Fine tune the model:
First run finale_project_just_train_shulhan_aruch.py

After that run:
finale_project_just_train.py

# Eval creation:
## For the fine tuned model eval csv creation run:
finale_project_create_eval_csv.py
## For the RAG run the following:
RAG_llamma-2 create_eval_csv.py

# For the evaluation run the following:
finale_project_just_eval.py

# For just Rav talk run:
## RAG
RAG_llamma-2_Rav_talk.py

## Fine tuned:
finale_project_Rav_talk.py