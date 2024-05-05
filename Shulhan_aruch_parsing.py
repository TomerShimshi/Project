

import requests
from bs4 import BeautifulSoup
import codecs
import re
import csv
import re
import os

def remove_lines_with_substring(input_str, substring):
    lines = input_str.split('\n')  # Split the input string into lines
    #filtered_lines = [line for line in lines if not line.endswith(substring)]  # Filter out lines that end with the given substring
    filtered_lines = [line for line in lines if substring not in line]  # Filter out lines that end with the given substring
    output_str = '\n'.join(filtered_lines)  # Join the filtered lines back into a string
    return output_str

def parse_string_by_substring(input_str, substring):
    pattern = re.compile(rf'(?<={re.escape(substring)}).*?(?={re.escape(substring)})', re.DOTALL)
    parsed_output = '\n'.join(pattern.findall(input_str))
    return parsed_output
def find_lines_starting_with_substring(input_str, starting_substring):
    lines = input_str.split('\n')  # Split the input string into lines
    matching_lines = [index + 1 for index, line in enumerate(lines) if line.startswith(starting_substring)]  # Enumerate through lines and find those that start with the given substring
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
    full_text_path = "kitzur_shulhan_aruch.txt"
    full_text =  open(full_text_path, "r", encoding='utf-8') 
    full_text = full_text.read()
    full_text = remove_lines_with_substring(full_text,substring = "/967")
    full_text = remove_lines_with_substring(full_text,substring = "1:11 PM")
    full_text = remove_lines_with_substring(full_text,substring = "lang=en")
    full_text = remove_lines_with_substring(full_text,substring = "‬")
    locations = find_lines_starting_with_substring(full_text, "Siman") 
    chapters = pars_full_text(full_text,lines_locations= locations)
    dataset = {'text':chapters}
    modified_file_path_txt = "sorted_kitzur_shulhan_aruch.txt"
    csv_path = modified_file_path_txt.replace(".txt",'.csv')
    #
    for chapter in chapters:
        with codecs.open(modified_file_path_txt, 'a', encoding='utf-8') as output_file:
                        output_file.write(chapter)
        dataset = {'text':chapter}
        append_dict_to_csv(dictionary=dataset,filename=csv_path)