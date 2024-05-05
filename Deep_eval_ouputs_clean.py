

import requests
from bs4 import BeautifulSoup
import json
import re
import csv
import re
import os
import yaml

def remove_lines_with_substring(input_str, substring,split_by:str = "\n"):
    lines = input_str.split(split_by)  # Split the input string into lines
    filtered_lines = [line for line in lines if substring not in line ]  # Filter out lines that end with the given substring
    output_str = split_by.join(filtered_lines)  # Join the filtered lines back into a string
    return output_str

def parse_string_by_substring(input_str, substring):
    pattern = re.compile(rf'(?<={re.escape(substring)}).*?(?={re.escape(substring)})', re.DOTALL)
    parsed_output = '\n'.join(pattern.findall(input_str))
    return parsed_output
def find_location_starting_with_substring(input_str, substring ,split_by:str = "\n"):
    lines = input_str.split(split_by)  # Split the input string into lines
    matching_lines = [index + 1 for index, line in enumerate(lines) if substring in line]  # Enumerate through lines and find those that start with the given substring
    return matching_lines

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
    full_text_path ="dataset\\20240504_173218.json" # "dataset\\20240504_190145.json" #"dataset\\20240504_173218.json"
    full_text =  open(full_text_path, "r", encoding='utf-8') 
    full_text = full_text.read()
    full_text = full_text.replace('false','False')
    full_text = full_text.replace('true','True')
    dict1= eval(full_text)
    #full_text = remove_lines_with_substring(full_text,substring = "\\u", split_by= " ")
    #tests_locations = find_location_starting_with_substring(input_str= full_text,substring='name\\',split_by=' ')
    save_path_yaml = "dataset\sorted_deep_eval_output_new.yaml"
    save_path_csv = save_path_yaml.replace('yaml','csv')
    #with open(save_path, 'w', encoding='utf-8') as f:
    #    json.dump(full_text, f)
    
    # read the json file into a dictionary
    for test in dict1['testCases']:
        append_dict_to_csv(test,save_path_yaml)
    with open(save_path_yaml, 'w') as outfile:
        yaml.dump(dict1['testCases'], outfile, default_flow_style=False)
    t=1
    