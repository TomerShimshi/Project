

import requests
from bs4 import BeautifulSoup
import json
import re
import csv
import re
import os
import yaml
import numpy as np



def pars_full_text(input_str:str, lines_locations:list):
    all_lines = input_str.split('\n')
    parsed_chapters =[]
    for i in range(len(lines_locations)-1):
        chapter_lines = all_lines[lines_locations[i]:lines_locations[i+1]]
        chapter = '\n'.join(chapter_lines)
        chapter = remove_lines_with_substring(chapter,"Siman")
        parsed_chapters.append(chapter)
    return parsed_chapters

def append_dict_to_csv(dictionary, filename):
    if not os.path.exists(filename):
        char = 'w'
    else:
        char = 'a'
        
    with open(filename, char, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(dictionary)



if __name__ == "__main__":
    # first we load the text files
    full_text_path ="test_eval_save_new_with_eval_steps_gpt-3.5-turbo.yaml" # "dataset\\20240504_190145.json" #"dataset\\20240504_173218.json"
    with open(full_text_path) as f:
        deep_eval_dict = yaml.load(f, Loader=yaml.FullLoader)
    scores = [item['Correctness (GEval)']['score'] for item in deep_eval_dict]
    mean_score = np.mean(scores)
    print(f"the corctness score is {mean_score}")
    
    
    
    
  
    