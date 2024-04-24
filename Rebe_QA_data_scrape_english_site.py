

import requests
from bs4 import BeautifulSoup
import codecs
import re
import csv
import time
import os

def get_Q_and_A_from_text(text):
    paragraphs = re.split(r'\n+', text.strip())

    # Initialize variables to store questions and answers
    #question =re.search(r'^(.*\n)?[^\n?]*\?', text, re.MULTILINE).group(0).strip()
    
    answer_started = False
    question_started = False
    answer_lines = []
    question_lines = []
    

    # Iterate through paragraphs to identify questions and answers
    for paragraph in paragraphs:
       
        if not answer_started and question_started:
            # Look for the start of the answer
            if "Answer:" in paragraph :
                answer_started = True
                question_started = False
                temp_paragraph = paragraph.replace("Answer:", '', 1)
                answer_lines.append(temp_paragraph)
        elif answer_started:
            if 'Sources:' in paragraph or 'PREVIOUS QUESTION' in paragraph or '#' in paragraph:
             break
            else:
                # Collect lines as part of the answer
                answer = paragraph.replace('\xa0','')
                if len(answer)>0:
                    answer_lines.append(paragraph)
        if not question_started and not answer_started:
            if "?" in paragraph and not question_started:
                question_started = True
                question_lines.append(paragraph)
                if "Answer:" not in text:
                    answer_started = True
                    
        elif not answer_started:
            question = paragraph.replace('\xa0','')
            if len(question)>0 and question != 'Question:':
                question_lines.append(question)
        
    # Join answer lines to form the answer
    answer = "\n".join(answer_lines)
    question = "\n".join(question_lines)
    question = question.replace("'",'')
    answer = answer.replace("'",'')
    return question, answer

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


if __name__ == "__main__":
    # Replace this URL with the URL of the Hebrew web page you want to scrape
    output_file_path = "Rebe_Q_and_A_dataset_just_rebe_questions_english.txt"
    csv_output_file = output_file_path.replace(".txt",".csv")
    num_of_questions = 0
    for i in range(1,1000000):
        #we stoped here 53288
        #time.sleep(0.25)
        url = f"https://asktherav.com/{i}/"

        # Send an HTTP request to the web page
        try:
            response = requests.get(url)
        except:
            continue

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the text content from the page
            page_class = soup.find_all("div", class_="col-md-8")
            page_text = page_class[0].get_text() #""
            if i ==10:
               continue
           
            
            question, answer = get_Q_and_A_from_text(page_text)
            if len(question) > 1 and len(answer) > 1 and "שאלה:" not in page_text and "תשובה:" not in page_text and has_english_letter(question + answer): #and has_english_letter(answer):
                save_txt = f"###question \n {question}.\n ###answer \n {answer}"

                # Save the text to a file with proper encoding for Hebrew (utf-8)
                with codecs.open(output_file_path, 'a', encoding='utf-8') as output_file:
                    output_file.write(save_txt)
                Q_A_dict = {"question":question, "answer": answer,"text":save_txt}
                append_dict_to_csv(Q_A_dict, csv_output_file)

                print(f"Text saved to {output_file_path} saved question num {i}")
                num_of_questions += 1
            else:

                print(f"no QA was found for idx {i}")
        else:
            print(f"Failed to retrieve the web page. Status code: {response.status_code}")
    print(f"total number of questions saved:{num_of_questions}")