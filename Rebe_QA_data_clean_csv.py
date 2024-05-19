

import requests
from bs4 import BeautifulSoup
import codecs
import re
import csv
import time
import os
import unicodedata

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



def has_english_letter(s):
    for char in s:
        if char.isalpha(): #and char.isascii():
            return True
    return False
def remove_hebrew_letters(input_string):
    letters = ['א','ב','ג','ד','ה','ו','ז','ח','ט','י','כ','ל','מ','נ','ס','ע','פ','צ','ק','ר','ש','ת','ך','ף','ם','ן']
    removed_letter = False
    for letter in letters:
        if letter in input_string:
            #input_string = input_string.replace(letter,"")
            #test = [char for char in input_string if char.isalpha()]
            #if len(test)<=1:
            return ""
            
            
    return input_string


if __name__ == "__main__":
    # Replace this URL with the URL of the Hebrew web page you want to scrape
    data_file_path = "Rebe_Q_and_A_dataset_just_rebe_questions_english_no_hebrew_v2.csv"
    save_path =  "cleaned_Rebe_Q_and_A_dataset_just_rebe_questions_english_no_hebrew_v2.csv"
    question_dict = csv_to_dict(data_file_path)
    cleaned_dict = {k:[] for k in question_dict.keys()}
    for i in range(len(question_dict['question'])):
        if question_dict['question'][i] not in cleaned_dict['question']:
            for k in cleaned_dict.keys():
                cleaned_dict[k].append(question_dict[k][i])
    for i in range(len(cleaned_dict['question'])):
        temp_dict ={k:cleaned_dict[k][i] for k in cleaned_dict.keys()}
        append_dict_to_csv(temp_dict,save_path)
    print(f"reduced the Q & A from {len(question_dict['question'])} to {len(cleaned_dict['question'])}")
    t=1