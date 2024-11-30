import datetime
import itertools
from itertools import combinations
import csv
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
import base64
import re
import pandas as pd
import scipy.stats as scipy_stats
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Replace all file paths with your own file paths where neccesary
# also dont forget to change the openAI key in the chatgpt_function
#-----------------------------------------------------------------

def create_deck():
    '''
    Create all combinations of cards for a classic Set game
    '''
    colors = ["Red", "Green", "Blue"]
    shapes = ["Ellipse", "Squiggle", "Diamond"]
    numbers = ["1", "2", "3"]
    shadings = ["Filled", "Lines", "Empty"]
    deck = [(color, shape, number, shading) for color in colors for shape in shapes for number in numbers for shading in shadings]
    return deck
def create_pos(deck, num_cards=12):
    return random.sample(deck, num_cards)
def cards_to_string(position):
    '''
    Turns a position of cards into a string to be fed into chatGPT
    '''
    s = ""
    for card in position:
        for attribute in card:
            s += attribute + ", "
        s=s[:-2]
        s += "\n"
    return s

def is_set(triplet):
    for i in range(4):  # Check each of the four positions
        counts = {}
        for quad in triplet:
            if quad[i] in counts:
                counts[quad[i]] += 1
            else:
                counts[quad[i]] = 1
        # If any position has exactly two of the same values, it's invalid
        if 2 in counts.values():
            return False
    return True

def chatgpt_function(prompt, tokens=4000, model='gpt-3.5-turbo', temp=0.1):
    llm = ChatOpenAI(model_name=model, temperature=temp, max_tokens=tokens, openai_api_key='OPENAI KEY')
    chatgpt_output = llm.invoke(prompt)
    return chatgpt_output

def trim_key(text):
    blocks = text.strip().split("\n\n")
    result = []
    for block in blocks:
        lines = block.split("\n")
        line_symbols = []
        for line in lines:
            components = [component.strip() for component in line.split(',')]
            first_letters = ''.join(component[0] for component in components if component)
            line_symbols.append(first_letters)
        result.append(line_symbols)
    
    return result

def trim_answer(input_text):
    lines = input_text.strip().split('\n')
    result = []
    for line in lines:
        line=line[3:]
        parts = [part.strip() for part in line.split(',')]
        first_chars = [part[0] for part in parts]
        result.append(''.join(first_chars))
    return result
def trim_answer2(text):
# Define the words to look for and their corresponding symbols
    words_to_symbols = {
    "Red": "R",
    "Blue": "B",
    "Green": "G",
    "Squiggle": "S",
    "Diamond": "D",
    "Ellipse": "E",
    "1": "1",
    "2": "2",
    "3": "3",
    "Filled": "F",
    "Striped": "S",
    "Empty": "E",
    "Lines": "L"}
    categories = {
    "color": ["Red", "Blue", "Green"],
    "shape": ["Squiggle", "Diamond", "Ellipse"],
    "number": ["1", "2", "3"],
    "fill": ["Filled", "Empty", "Lines"]
}    
    # Remove leading numbers and parentheses if present
    text = re.sub(r'^\d+\s*(?:\([^)]+\))?', '', text, flags=re.MULTILINE)
    
    # Use a regex to extract the relevant words
    pattern = re.compile(r'\b(Red|Blue|Green|Squiggle|Diamond|Ellipse|1|2|3|Filled|Striped|Empty|Lines)\b', re.IGNORECASE)
    
    matches = pattern.findall(text)
    
    result = []
    card_details = []
    
    if matches:
        card_count = 0
        
        for match in matches:
            for category, words in categories.items():
                if match.capitalize() in words:
                    card_details.append(match.capitalize())
                    break
            
            # Group matches in sets of 4 to form each card's details
            if len(card_details) == 4:
                # Order the details by the predefined categories
                ordered_details = [
                    next(detail for detail in card_details if detail in categories['color']),
                    next(detail for detail in card_details if detail in categories['shape']),
                    next(detail for detail in card_details if detail in categories['number']),
                    next(detail for detail in card_details if detail in categories['fill'])
                ]
                symbol_string = ''.join(words_to_symbols[word] for word in ordered_details)
                result.append(symbol_string)
                card_details = []
                card_count += 1
            
            if card_count == 3:
                break
    
    return result

def check_permutations(symbols, list_of_lists):
    all_permutations = list(itertools.permutations(symbols))
    for permutation in all_permutations:
        if list(permutation) in list_of_lists:
            return "1"
    return "0"

def get_variance(set):
    try:
        variance = 0
        num_positions = len(set[0])  # Assuming all strings have the same length

        for i in range(num_positions):
            chars = {set[0][i], set[1][i], set[2][i]}
            if len(chars) > 1:
                variance += 1
        return variance
    except:
        return 'nan'

def set_count_key(text):
    # Function to count '[' symbols in a block
    count = 0
    in_block = False
    for char in text:
        if char == '[':
            in_block = True
        elif char == ']' and in_block:
            in_block = False
            count += 1
    return count

def edit_csv():
    folder_path = 'Thesis code\\Data\\Default_GPT3.5_n100' #replace with your own file path
    answers = {}
    answerslength = {}
    for filename in os.listdir(folder_path):    
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='cp1252') as file:
                capture = False
                text_buffer = []
                for line in file:
                    if 'REDUCED ANSWER' in line and capture:
                        capture = False
                        break
                    if capture:
                        text_buffer.append(line)
                    if 'ANSWER' in line:
                        capture = True
                text = ''.join(text_buffer) + '\n'
                n_attempts_prompt = '''I am testing participants' ability to find Sets.
                The following is an answer given by a participant. Just respond
                to this message with the amount of attempts done by the participant. Explain your reasoning. \n\nPARTICIPANT ANSWER\n\n''' + text
                n_attempts=chatgpt_function(prompt=n_attempts_prompt, temp=0.5)
                n_attempts_number=chatgpt_function(prompt="reduce this text until there is only the amount of attempts left in a number. Answer with only a number.\n\n"+n_attempts.content, temp=0.1)
                print(n_attempts_number.content)
                answers[int(filename[29:][:-4])]=n_attempts_number.content
                answerslength[int(filename[29:][:-4])]=len(text)
    print(answers)
    input_file='Thesis code\\Data\\Default_GPT3.5_n100\\data_2024-05-12_12-34-53.csv'
    output_file='Thesis code\\Data\\Default_GPT3.5_n100\\data_2024-05-12_12-34-53-updated.csv'
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            for row in reader:
                try:
                    row.append(answers[int(row[1])])  # Add count to 12th column
                    row.append(answerslength[int(row[1])])
                    row.append(answers[int(row[1])]/answerslength[int(row[1])])
                    writer.writerow(row)
                except:
                    pass

def all_sets():
    deck=create_deck()
    all_combs=list(combinations(deck,3))
    all_sets={0:[],1:[], 2:[], 3:[], 4:[]}
    for comb in all_combs:
        if is_set(comb):
            all_sets[get_variance(comb)].append(comb)
        else:
            all_sets[0].append(comb)
    return all_sets

def replace_words(text, mapping):
    # Replace each key in the text with its corresponding value
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text
def transform_tuple(tpl, mapping):
    return tuple(mapping.get(item, item) for item in tpl)
def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
#----------------------------------------------------------------- TESTS

def GPT_default(subject_prompt, test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.9, eval_temp=0.1, iter=1, dataset_name='Default_unnamed'):
    
    data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Attempts', 'Tokens', 'Trimmed Answer', 'Avg Key Var', 'Answer Var', 'Var diff', 'Self-eval', 'Key-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
    current_datetime = datetime.datetime.now()
    filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'Thesis code\\Data\\{dataset_name}\\data_{filename_datetime}.csv' #replace with your own file path
    directory_path=f'Thesis code\\Data\\{dataset_name}' #replace with your own file path
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"Directory '{directory_path}' already exists.")
        return False
    with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    while iter > 0:
        deck = create_deck()
        foundset=0
        while(foundset==0): # we want there to actually be a Set available
            pos = create_pos(deck)
            all_combs = list(combinations(pos, 3))
            sets = [set for set in all_combs if is_set(set)]
            nl_key=''
            for s in sets:
                nl_key+=cards_to_string(s)
                nl_key+="\n"
                foundset=1
        pos_string = cards_to_string(pos)
        trimmed_pos= trim_key(pos_string)
        trimmed_key=trim_key(nl_key)
        avg_key_var=0
        for set in trimmed_key:
            avg_key_var+=get_variance(set)
        avg_key_var=avg_key_var / len(trimmed_key)        
    
        print("---------Question---------")
        set_prompt=subject_prompt + pos_string
        print(set_prompt + "\n")


        print("---------NL Key---------")
        print(nl_key)
        print("---------Trimmed Key---------")
        print(trimmed_key)

            
        print("---------Response---------")
        answer=chatgpt_function(prompt=set_prompt, model=test_model, temp=test_temp)
        print(answer.content)

        print("\n---------Reduced---------\n")
        reduced_prompt='''In the card game Set, three cards are a Set if for each property they have, 
        all three cards are all the same or all different. I am testing participants' ability to find Sets.
        The following is an answer given by a participant. It might contain reasoning or a thought process. 
        I want you to  remove all failed attempts at finding sets and remove all reasoning. Instead, just respond
        to this message with the last three cards given by the participant. Use the following format:
        \n\n
        1. Color, Shape, Number, Filling\n
        2. Color, Shape, Number, Filling\n
        3. Color, Shape, Number, Filling\n.\n\nPARTICIPANT ANSWER\n\n''' + answer.content
        reduced_answer=chatgpt_function(prompt=reduced_prompt, model=eval_model, temp=eval_temp)
        print(reduced_answer.content)

        

        print("\n---------Self-Evaluation---------\n")
        evaluation_prompt=f'''I am testing participants' ability to find Sets. The following is an answer
        given by a participant. Look at their answer; did the participant THINK they found a correct set? 
        To find this, mainly look at the last few sentences of their answer.
        Respond to this message with your reasoning, then end your message with "1" if the participant thinks they found a correct set,
        or "0" if they think they didn't find the correct answer.\n\nPARTICIPANT ANSWER\n\n{answer.content}'''
        evaluation=chatgpt_function(prompt=evaluation_prompt, model=eval_model, temp=eval_temp)
        self_eval=(evaluation.content[-1])
        if (self_eval=='.'):
            self_eval = evaluation.content[-2]
        print(evaluation.content)

        print("\n---------Trimmed---------\n")
        #COMMENTED OUT BECAUSE IT'S NOT ACCURATE ENOUGH
        #trim_prompt='''I am processing natural language into usable text for data processing.
        #The following is an answer given by a participant in my research. It contains three cards with four attributes,
        #like in the card game Set. I want you to match the participant's answer to the answer key and ONLY respond with the
        #set as given in the answer key, if it is present. In the answer key, attributes are named only by their first letter.
        #\n\nPARTICIPANT ANSWER\n\n''' + answer.content + '''\n\nANSWER KEY\n\n''' + answer_key
        #trimmed=chatgpt_function(prompt=trim_prompt, model=eval_model, temp=eval_temp)
        #print(trimmed.content)
        try:
            trimmed=trim_answer2(reduced_answer.content)
        except:
            try:
                trimmed=trim_answer(reduced_answer.content)
            except: 
                trimmed='Illegible'
        print(trimmed)

        print("\n---------Key-Evaluation---------\n")
        key_eval=check_permutations(trimmed, trimmed_key)
        print(key_eval)

        n_attempts_prompt = '''I am testing participants' ability to find Sets.
        The following is an answer given by a participant. Just respond
        to this message with the amount of attempts done by the participant. Explain your reasoning. \n\nPARTICIPANT ANSWER\n\n''' + answer.content
        n_attempts=chatgpt_function(prompt=n_attempts_prompt, temp=0.5)
        n_attempts_number=chatgpt_function(prompt="Reduce this text until there is only the amount of attempts left as a number. Answer with only a number.\n\n"+n_attempts.content, temp=0.1)
        attempts=int(n_attempts_number.content)

        print("Writing data...\n\n")
        
        current_datetime = datetime.datetime.now()
        filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        answervar=get_variance(trimmed)
        try:
            vardif = answervar-avg_key_var
        except:
            vardif = 'Illegible'
        new_data_row=[filename_datetime, iter, trimmed_pos, trimmed_key, len(trimmed_key), attempts, len(answer.content), trimmed, avg_key_var, answervar, vardif, self_eval, key_eval, test_temp, eval_temp, test_model, eval_model]
        #data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Attempts', 'Tokens', 'Trimmed Answer', 'Avg Key Var', 'Answer Var', 'Var diff', 'Self-eval', 'Key-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
        data.append(new_data_row)
        with open(f'Thesis code\\Data\\{dataset_name}\\response-{filename_datetime}-{str(iter)}.txt', 'w', encoding="utf-8") as file:
            file.write(f'ANSWER\n\n{answer.content}\n\nREDUCED ANSWER\n\n{reduced_answer.content}')
        iter-=1
        with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_data_row)
    print('Done')

def GPT_Reasoning_Comparison(subject_prompt, test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.9, eval_temp=0.1, iter=1, dataset_name='Default_unnamed'):
    
    data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Trimmed Answer C', 'Trimmed AnswerC ', 'Avg Key Var', 'Answer VarC', 'Var diffC', 'Answer VarV', 'Var diffV','Key-eval-C', 'Key-eval-V', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
    current_datetime = datetime.datetime.now()
    filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'Thesis code\\Data\\{dataset_name}\\data_{filename_datetime}.csv'
    directory_path=f'Thesis code\\Data\\{dataset_name}'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"Directory '{directory_path}' already exists.")
        return False
    with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    while iter > 0:
        deck = create_deck()
        foundset=0
        while(foundset==0): # we want there to actually be a Set available
            pos = create_pos(deck)
            all_combs = list(combinations(pos, 3))
            sets = [set for set in all_combs if is_set(set)]
            nl_key=''
            for s in sets:
                nl_key+=cards_to_string(s)
                nl_key+="\n"
                foundset=1
        pos_string = cards_to_string(pos)
        trimmed_pos= trim_key(pos_string)
        trimmed_key=trim_key(nl_key)
        avg_key_var=0
        for set in trimmed_key:
            avg_key_var+=get_variance(set)
        avg_key_var=avg_key_var / len(trimmed_key)        
    
        print("---------Question---------")
        set_prompt=subject_prompt + pos_string
        print(set_prompt + "\n")
        set_prompt2 = f'''In the card game Set, three cards are a Set if for each property they have, all three cards are all the same or all different.
        Name a combination of three cards that forms a valid Set from these 12 cards, and include reasoning as to why you chose those cards. If it turns out you were incorrect
        in your choice and the cards actually don't make a Set, do not try again, just end your response. Use the same format as the cards given:\n\n{pos_string}'''

        print("---------NL Key---------")
        print(nl_key)
        print("---------Trimmed Key---------")
        print(trimmed_key)

            
        print("---------Response---------")
        answer=chatgpt_function(prompt=set_prompt, model=test_model, temp=test_temp)
        print(answer.content)
        
        answer2=chatgpt_function(prompt=set_prompt2, model=test_model, temp=test_temp)
        print(answer2.content)

        print("\n---------Reduced---------\n")
        reduced_prompt='''In the card game Set, three cards are a Set if for each property they have, 
        all three cards are all the same or all different. I am testing participants' ability to find Sets.
        The following is an answer given by a participant. It might contain reasoning or a thought process. 
        I want you to  remove all reasoning. Instead, just respond
        to this message with the three cards given by the participant, numbered one to three. Nothing else.\n\nPARTICIPANT ANSWER\n\n''' + answer.content
        reduced_prompt2='''In the card game Set, three cards are a Set if for each property they have, 
        all three cards are all the same or all different. I am testing participants' ability to find Sets.
        The following is an answer given by a participant. It might contain reasoning or a thought process. 
        I want you to  remove all reasoning. Instead, just respond
        to this message with the three cards given by the participant, numbered one to three. Nothing else.\n\nPARTICIPANT ANSWER\n\n''' + answer2.content
        reduced_answer=chatgpt_function(prompt=reduced_prompt, model=eval_model, temp=eval_temp)
        print(reduced_answer.content)
        reduced_answer2=chatgpt_function(prompt=reduced_prompt2, model=eval_model, temp=eval_temp)
        print(reduced_answer2.content)

        

        print("\n---------Trimmed---------\n")
        try:
            trimmed=trim_answer(reduced_answer.content)
        except:
            trimmed='Illegible'
        print(trimmed)

        try:
            trimmed2=trim_answer(reduced_answer2.content)
        except:
            trimmed2='Illegible'
        print(trimmed2)

        print("\n---------Key-Evaluation---------\n")
        key_eval=check_permutations(trimmed, trimmed_key)
        print(key_eval)

        key_eval2=check_permutations(trimmed2, trimmed_key)
        print(key_eval2)

        print("Writing data...\n\n")
        
        current_datetime = datetime.datetime.now()
        filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        answervar=get_variance(trimmed)
        try:
            vardif = answervar-avg_key_var
        except:
            vardif = 'Illegible'
        answervar2=get_variance(trimmed2)
        try:
            vardif2 = answervar2-avg_key_var
        except:
            vardif2 = 'Illegible'
            
        new_data_row=[filename_datetime, iter, trimmed_pos, trimmed_key, len(trimmed_key), trimmed, trimmed2, avg_key_var, answervar, vardif, answervar2, vardif2, key_eval, key_eval2, test_temp, eval_temp, test_model, eval_model]
        #data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Attempts', 'Tokens', 'Trimmed Answer', 'Avg Key Var', 'Answer Var', 'Var diff', 'Self-eval', 'Key-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
        data.append(new_data_row)
        with open(f'Thesis code\\Data\\{dataset_name}\\response-{filename_datetime}-{str(iter)}.txt', 'w', encoding="utf-8") as file:
            file.write(f'ANSWER1\n\n{answer.content}\n\nREDUCED ANSWER1\n\n{reduced_answer.content}\n\nANSWER2\n\n{answer2.content}\n\nREDUCED ANSWER2\n\n{reduced_answer2.content}')
        iter-=1
        with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_data_row)
    print('Done')

def GPT_classification(test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.9, eval_temp=0.3, iter=1, dataset_name='Classification_unnamed'):
#file initialization
    data = ['DateTime', 'Iter', 'Set', 'Key', 'Set Var', 'Tokens', 'Trimmed Answer', 'Self-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
    current_datetime = datetime.datetime.now()
    filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'Thesis code\\Data\\{dataset_name}\\data_{filename_datetime}.csv'
    directory_path=f'Thesis code\\Data\\{dataset_name}'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"Directory '{directory_path}' already exists.")
        return False
    with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

#set initialization
    maxiter=iter
    sets_by_var=all_sets()
    while iter > 0:
        if iter > maxiter/2:
            var=(iter%4)+1
            set=random.choice(sets_by_var[var])
            valid=True
        else:
            var='nan'
            set=random.choice(sets_by_var[0])
            valid=False
        set_nl=cards_to_string(set)
        print("---------Question---------")
        set_prompt='''In the card game Set, three cards are a Set if for each property they have, 
all three cards are all the same or all different. Do these three cards make a valid Set? Explain
your reasoning:\n\n''' + set_nl
        print(set_prompt + "\n")

        print("---------Key---------")
        print(valid)
            
        print("---------Response---------")
        answer=chatgpt_function(prompt=set_prompt, model=test_model, temp=test_temp)
        print(answer.content)

        print("\n---------Reduced---------\n")
        reduced_prompt='''The following is an answer given by a participant in a research about Set. It contains reasoning and 
        a statement about the validity of a Set. What does the participant think about the set it had to evaluate? Was it a valid set or not,
        according to the participant? Answer by summarizing the response, and then end your message with "0" if the participant thought the set
        was invalid and "1" if they thought it was valid. Make sure the number is the last symbol in your message.\n\nPARTICIPANT ANSWER\n\n''' + answer.content
        reduced_answer=chatgpt_function(prompt=reduced_prompt, model=eval_model, temp=eval_temp)
        
        print(reduced_answer.content)

        print("Writing data...\n\n")
        
        current_datetime = datetime.datetime.now()
        filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        new_data_row=[filename_datetime, iter, set, valid, var, len(answer.content), reduced_answer.content, reduced_answer.content[-1], test_temp, eval_temp, test_model, eval_model]
        #data = ['DateTime', 'Iter', 'Set', 'Key', 'Set Var', 'Tokens', 'Trimmed Answer', 'Self-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
        data.append(new_data_row)
        with open(f'Thesis code\\Data\\{dataset_name}\\response-{filename_datetime}-{str(iter)}.txt', 'w', encoding="utf-8") as file:
            file.write(f'ANSWER\n\n{answer.content}\n\nREDUCED ANSWER\n\n{reduced_answer.content}')
        print(f'Completed iteration {iter}')
        iter-=1
        with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_data_row)
        
    print('Done')            

def GPT_variability(subject_prompt, test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.9, eval_temp=0.1, iter=1, dataset_name='Variability_unnamed'):
    
    data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Attempts', 'Tokens', 'Trimmed Answer', 'Set1 Var', 'Set2 Var', 'Answer Var', 'Var diff', 'Self-eval', 'Key-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
    current_datetime = datetime.datetime.now()
    filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'Thesis code\\Data\\{dataset_name}\\data_{filename_datetime}.csv'
    directory_path=f'Thesis code\\Data\\{dataset_name}'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"Directory '{directory_path}' already exists.")
        return False
    with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    while iter > 0:
        deck = create_deck()
        foundset=0
        var=[]
        condition=False
        while(condition==False): #we want 2 specific sets
            var=[]
            foundset=0
            pos = create_pos(deck)
            all_combs = list(combinations(pos, 3))
            sets = [set for set in all_combs if is_set(set)]
            nl_key=''
            for s in sets:
                nl_key+=cards_to_string(s)
                nl_key+="\n"
                foundset+=1
                var.append(get_variance(s))
            if (foundset==2):
                if (var[0] != var[1]):
                    condition=True
                
        pos_string = cards_to_string(pos)
        trimmed_pos= trim_key(pos_string)
        trimmed_key=trim_key(nl_key)
        key_var=[]
        for set in trimmed_key:
            key_var.append(get_variance(set))  
    
        print("---------Question---------")
        set_prompt=subject_prompt + pos_string
        print(set_prompt + "\n")


        print("---------NL Key---------")
        print(nl_key)
        print("---------Trimmed Key---------")
        print(trimmed_key)

            
        print("---------Response---------")
        answer=chatgpt_function(prompt=set_prompt, model=test_model, temp=test_temp)
        print(answer.content)

        print("\n---------Reduced---------\n")
        reduced_prompt='''In the card game Set, three cards are a Set if for each property they have, 
        all three cards are all the same or all different. I am testing participants' ability to find Sets.
        The following is an answer given by a participant. It might contain reasoning or a thought process. 
        I want you to  remove all failed attempts at finding sets and remove all reasoning. Instead, just respond
        to this message with the last three cards given by the participant, numbered one to three. Nothing else.\n\nPARTICIPANT ANSWER\n\n''' + answer.content
        reduced_answer=chatgpt_function(prompt=reduced_prompt, model=eval_model, temp=eval_temp)
        print(reduced_answer.content)

        

        print("\n---------Self-Evaluation---------\n")
        evaluation_prompt=f'''I am testing participants' ability to find Sets. The following is an answer
        given by a participant. Look at their answer; did the participant THINK they found a correct set? 
        To find this, mainly look at the last few sentences of their answer.
        Respond to this message with your reasoning, then end your message with "1" if the participant thinks they found a correct set,
        or "0" if they think they didn't find the correct answer. Make sure the number is the last symbol in your message.\n\nPARTICIPANT ANSWER\n\n{answer.content}'''
        evaluation=chatgpt_function(prompt=evaluation_prompt, model=eval_model, temp=eval_temp)
        self_eval=(evaluation.content[-1])
        if (self_eval=='.'):
            self_eval = evaluation.content[-2]
        print(evaluation.content)

        print("\n---------Trimmed---------\n")
        #trim_prompt='''I am processing natural language into usable text for data processing.
        #The following is an answer given by a participant in my research. It contains three cards with four attributes,
        #like in the card game Set. I want you to match the participant's answer to the answer key and ONLY respond with the
        #set as given in the answer key, if it is present. In the answer key, attributes are named only by their first letter.
        #\n\nPARTICIPANT ANSWER\n\n''' + answer.content + '''\n\nANSWER KEY\n\n''' + answer_key
        #trimmed=chatgpt_function(prompt=trim_prompt, model=eval_model, temp=eval_temp)
        #print(trimmed.content)
        try:
            trimmed=trim_answer(reduced_answer.content)
        except:
            trimmed='Illegible'
        print(trimmed)

        print("\n---------Key-Evaluation---------\n")
        key_eval=check_permutations(trimmed, trimmed_key)
        print(key_eval)

        n_attempts_prompt = '''I am testing participants' ability to find Sets.
        The following is an answer given by a participant. Just respond
        to this message with the amount of attempts done by the participant. Explain your reasoning. \n\nPARTICIPANT ANSWER\n\n''' + answer.content
        n_attempts=chatgpt_function(prompt=n_attempts_prompt, temp=0.5)
        n_attempts_number=chatgpt_function(prompt="Reduce this text until there is only the amount of attempts left as a number. Answer with only a number.\n\n"+n_attempts.content, temp=0.1)
        attempts=int(n_attempts_number.content)

        print("Writing data...\n\n")
        
        current_datetime = datetime.datetime.now()
        filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        answervar=get_variance(trimmed)
        try:
            vardif = answervar-key_var
        except:
            vardif = 'Illegible'
        new_data_row=[filename_datetime, iter, trimmed_pos, trimmed_key, len(trimmed_key), attempts, len(answer.content), trimmed, key_var[0], key_var[1], answervar, vardif, self_eval, key_eval, test_temp, eval_temp, test_model, eval_model]
        #data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Attempts', 'Tokens', 'Trimmed Answer', 'Avg Key Var', 'Answer Var', 'Var diff', 'Self-eval', 'Key-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
        data.append(new_data_row)
        with open(f'Thesis code\\Data\\{dataset_name}\\response-{filename_datetime}-{str(iter)}.txt', 'w', encoding="utf-8") as file:
            file.write(f'ANSWER\n\n{answer.content}\n\nREDUCED ANSWER\n\n{reduced_answer.content}')
        iter-=1 
        with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_data_row)
    print('Done')


def GPT_N_cards(subject_prompt, test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.9, eval_temp=0.1, iter=1, dataset_name='N_cards_unnamed', n=12):
    
    data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Card count', 'Attempts', 'Tokens', 'Trimmed Answer', 'Avg Key Var', 'Answer Var', 'Var diff', 'Self-eval', 'Key-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
    current_datetime = datetime.datetime.now()
    filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'Thesis code\\Data\\4-11 cards onetry\\{dataset_name}\\data_{filename_datetime}.csv'
    directory_path=f'Thesis code\\Data\\4-11 cards onetry\\{dataset_name}'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"Directory '{directory_path}' already exists.")
        return False
    with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    while iter > 0:
        deck = create_deck()
        foundset=0
        attempt=0
        while(foundset!=1): # we want there to be 1 Set available
            foundset=0
            attempt+=1
            print(f"Creating position, attempt {attempt}")
            pos = create_pos(deck, num_cards=n)
            all_combs = list(combinations(pos, 3))
            sets = [set for set in all_combs if is_set(set)]
            nl_key=''
            for s in sets:
                nl_key+=cards_to_string(s)
                nl_key+="\n"
                foundset+=1
            
        pos_string = cards_to_string(pos)
        trimmed_pos= trim_key(pos_string)
        trimmed_key=trim_key(nl_key)
        avg_key_var=0
        for set in trimmed_key:
            avg_key_var+=get_variance(set)
        avg_key_var=avg_key_var / len(trimmed_key)        
    
        print("---------Question---------")
        set_prompt=subject_prompt + pos_string
        print(set_prompt + "\n")


        print("---------NL Key---------")
        print(nl_key)
        print("---------Trimmed Key---------")
        print(trimmed_key)

            
        print("---------Response---------")
        answer=chatgpt_function(prompt=set_prompt, model=test_model, temp=test_temp)
        print(answer.content)

        print("\n---------Reduced---------\n")
        reduced_prompt='''In the card game Set, three cards are a Set if for each property they have, 
        all three cards are all the same or all different. I am testing participants' ability to find Sets.
        The following is an answer given by a participant. It might contain reasoning or a thought process. 
        I want you to  remove all failed attempts at finding sets and remove all reasoning. Instead, just respond
        to this message with the last three cards given by the participant. Follow this format:
        \n\n1. Color, Shape, Number, Filling\n2. Color, Shape, Number, Filling\n3. Color, Shape, Number, Filling\n\nPARTICIPANT ANSWER\n\n''' + answer.content
        reduced_answer=chatgpt_function(prompt=reduced_prompt, model=eval_model, temp=eval_temp)
        print(reduced_answer.content)

        

        print("\n---------Self-Evaluation---------\n")
        evaluation_prompt=f'''I am testing participants' ability to find Sets. The following is an answer
        given by a participant. Look at their answer; did the participant THINK they found a correct set? 
        To find this, mainly look at the last few sentences of their answer.
        Respond to this message with your reasoning, then end your message with "1" if the participant thinks they found a correct set,
        or "0" if they think they didn't find the correct answer. Make sure the number is the last symbol in your message.\n\nPARTICIPANT ANSWER\n\n{answer.content}'''
        evaluation=chatgpt_function(prompt=evaluation_prompt, model=eval_model, temp=eval_temp)
        self_eval=(evaluation.content[-1])
        if (self_eval=='.'):
            self_eval = evaluation.content[-2]
        
        print(evaluation.content)

        print("\n---------Trimmed---------\n")
        #trim_prompt='''I am processing natural language into usable text for data processing.
        #The following is an answer given by a participant in my research. It contains three cards with four attributes,
        #like in the card game Set. I want you to match the participant's answer to the answer key and ONLY respond with the
        #set as given in the answer key, if it is present. In the answer key, attributes are named only by their first letter.
        #\n\nPARTICIPANT ANSWER\n\n''' + answer.content + '''\n\nANSWER KEY\n\n''' + answer_key
        #trimmed=chatgpt_function(prompt=trim_prompt, model=eval_model, temp=eval_temp)
        #print(trimmed.content)
        try:
            trimmed=trim_answer(reduced_answer.content)
        except:
            trimmed='Illegible'
        print(trimmed)

        print("\n---------Key-Evaluation---------\n")
        key_eval=check_permutations(trimmed, trimmed_key)
        print(key_eval)

        n_attempts_prompt = '''I am testing participants' ability to find Sets.
        The following is an answer given by a participant. Just respond
        to this message with the amount of attempts done by the participant. Explain your reasoning. \n\nPARTICIPANT ANSWER\n\n''' + answer.content
        n_attempts=chatgpt_function(prompt=n_attempts_prompt, temp=0.5)
        n_attempts_number=chatgpt_function(prompt="Reduce this text until there is only the amount of attempts left as a number. Answer with only a number.\n\n"+n_attempts.content, temp=0.1)
        attempts=int(n_attempts_number.content)

        print("Writing data...\n\n")
        
        current_datetime = datetime.datetime.now()
        filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        answervar=get_variance(trimmed)
        try:
            vardif = answervar-avg_key_var
        except:
            vardif = 'Illegible'
        new_data_row=[filename_datetime, iter, trimmed_pos, trimmed_key, len(trimmed_key), n, attempts, len(answer.content), trimmed, avg_key_var, answervar, vardif, self_eval, key_eval, test_temp, eval_temp, test_model, eval_model]
        #data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Attempts', 'Tokens', 'Trimmed Answer', 'Avg Key Var', 'Answer Var', 'Var diff', 'Self-eval', 'Key-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
        data.append(new_data_row)
        with open(f'Thesis code\\Data\\4-11 cards onetry\\{dataset_name}\\response-{filename_datetime}-{str(iter)}.txt', 'w', encoding="utf-8") as file:
            file.write(f'ANSWER\n\n{answer.content}\n\nREDUCED ANSWER\n\n{reduced_answer.content}')
        iter-=1
        with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_data_row)
    print('Done')

def GPT_abstract(subject_prompt, test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.9, eval_temp=0.1, iter=1, dataset_name='Abstract_unnamed'):
    
    data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Attempts', 'Tokens', 'Trimmed Answer', 'Avg Key Var', 'Answer Var', 'Var diff', 'Self-eval', 'Key-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
    current_datetime = datetime.datetime.now()
    filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'Thesis code\\Data\\{dataset_name}\\data_{filename_datetime}.csv'
    directory_path=f'Thesis code\\Data\\{dataset_name}'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"Directory '{directory_path}' already exists.")
        return False
    with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    while iter > 0:
        deck = create_deck()
        foundset=0
        while(foundset==0): # we want there to actually be a Set available
            pos = create_pos(deck)
            all_combs = list(combinations(pos, 3))
            sets = [set for set in all_combs if is_set(set)]
            nl_key=''
            for s in sets:
                nl_key+=cards_to_string(s)
                nl_key+="\n"
                foundset=1

        mapping = {
        'Red': 'Raspberry',
        'Blue': 'Banana',
        'Green': 'Grape',
        'Ellipse': 'Eagle',
        'Diamond': 'Dolphin',
        'Squiggle': 'Snake',
        '1': 'Wheat',
        '2': 'Wood',
        '3': 'Brick',
        'Filled': 'France',
        'Empty': 'Egypt',
        'Lines': 'Lapland'
        }
        reverse_mapping = {
        'Raspberry': 'Red',
        'Banana': 'Blue',
        'Grape': 'Green',
        'Eagle': 'Ellipse',
        'Dolphin': 'Diamond',
        'Snake': 'Squiggle',
        'Wheat': '1',
        'Wood': '2',
        'Brick': '3',
        'France': 'Filled',
        'Egypt': 'Empty',
        'Lapland': 'Lines'
        }
        trans_pos = [transform_tuple(item, mapping) for item in pos]


        pos_string = cards_to_string(trans_pos)
        trimmed_pos= trim_key(pos_string)
        trimmed_key=trim_key(nl_key)
        avg_key_var=0
        for set in trimmed_key:
            avg_key_var+=get_variance(set)
        avg_key_var=avg_key_var / len(trimmed_key)        
    
        print("---------Question---------")
        set_prompt=subject_prompt + pos_string
        print(set_prompt + "\n")


        print("---------NL Key---------")
        print(nl_key)
        print("---------Trimmed Key---------")
        print(trimmed_key)

            
        print("---------Response---------")
        answer=chatgpt_function(prompt=set_prompt, model=test_model, temp=test_temp)
        answer_content = replace_words(answer.content, reverse_mapping)
        print(answer_content)

        print("\n---------Reduced---------\n")
        reduced_prompt='''In the card game Set, three cards are a Set if for each property they have, 
        all three cards are all the same or all different. I am testing participants' ability to find Sets.
        The following is an answer given by a participant. It might contain reasoning or a thought process. 
        I want you to  remove all failed attempts at finding sets and remove all reasoning. Instead, just respond
        to this message with the last three cards given by the participant. Use the following format:
        \n\n
        1. Color, Shape, Number, Filling\n
        2. Color, Shape, Number, Filling\n
        3. Color, Shape, Number, Filling\n.\n\nPARTICIPANT ANSWER\n\n''' + answer_content
        reduced_answer=chatgpt_function(prompt=reduced_prompt, model=eval_model, temp=eval_temp)
        print(reduced_answer.content)

        

        print("\n---------Self-Evaluation---------\n")
        evaluation_prompt=f'''I am testing participants' ability to find Sets. The following is an answer
        given by a participant. Look at their answer; did the participant THINK they found a correct set? 
        To find this, mainly look at the last few sentences of their answer.
        Respond to this message with your reasoning, then end your message with "1" if the participant thinks they found a correct set,
        or "0" if they think they didn't find the correct answer.\n\nPARTICIPANT ANSWER\n\n{answer_content}'''
        evaluation=chatgpt_function(prompt=evaluation_prompt, model=eval_model, temp=eval_temp)
        self_eval=(evaluation.content[-1])
        if (self_eval=='.'):
            self_eval = evaluation.content[-2]
        print(evaluation.content)

        print("\n---------Trimmed---------\n")
        #trim_prompt='''I am processing natural language into usable text for data processing.
        #The following is an answer given by a participant in my research. It contains three cards with four attributes,
        #like in the card game Set. I want you to match the participant's answer to the answer key and ONLY respond with the
        #set as given in the answer key, if it is present. In the answer key, attributes are named only by their first letter.
        #\n\nPARTICIPANT ANSWER\n\n''' + answer.content + '''\n\nANSWER KEY\n\n''' + answer_key
        #trimmed=chatgpt_function(prompt=trim_prompt, model=eval_model, temp=eval_temp)
        #print(trimmed.content)
        try:
            trimmed=trim_answer2(reduced_answer.content)
        except:
            try:
                trimmed=trim_answer(reduced_answer.content)
            except: 
                trimmed='Illegible'
        print(trimmed)

        print("\n---------Key-Evaluation---------\n")
        key_eval=check_permutations(trimmed, trimmed_key)
        print(key_eval)

        n_attempts_prompt = '''I am testing participants' ability to find Sets.
        The following is an answer given by a participant. Just respond
        to this message with the amount of attempts done by the participant. Explain your reasoning. \n\nPARTICIPANT ANSWER\n\n''' + answer.content
        n_attempts=chatgpt_function(prompt=n_attempts_prompt, temp=0.5)
        n_attempts_number=chatgpt_function(prompt="Reduce this text until there is only the amount of attempts left as a number. Answer with only a number.\n\n"+n_attempts.content, temp=0.1)
        attempts=int(n_attempts_number.content)

        print("Writing data...\n\n")
        
        current_datetime = datetime.datetime.now()
        filename_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        answervar=get_variance(trimmed)
        try:
            vardif = answervar-avg_key_var
        except:
            vardif = 'Illegible'
        new_data_row=[filename_datetime, iter, trimmed_pos, trimmed_key, len(trimmed_key), attempts, len(answer_content), trimmed, avg_key_var, answervar, vardif, self_eval, key_eval, test_temp, eval_temp, test_model, eval_model]
        #data = ['DateTime', 'Iter', 'Position', 'Key', 'Set Count', 'Attempts', 'Tokens', 'Trimmed Answer', 'Avg Key Var', 'Answer Var', 'Var diff', 'Self-eval', 'Key-eval', 'Test_temp', 'Eval_temp', 'Test_model', 'Eval_model']
        data.append(new_data_row)
        with open(f'Thesis code\\Data\\{dataset_name}\\response-{filename_datetime}-{str(iter)}.txt', 'w', encoding="utf-8") as file:
            file.write(f'ANSWER\n\n{answer.content}\n\nREDUCED ANSWER\n\n{reduced_answer.content}')
        iter-=1
        with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_data_row)
    print('Done')

#------------------------------------------------------------------ DATA PROCESSING
def perform_anova_withindata(file_path, dependent='key-eval', independent1='Avg Key Var', independent2=None):
    # Load the data
    print(f"Testing {file_path}")
    data = pd.read_csv(file_path)
    data[dependent] = pd.to_numeric(data[dependent], errors='coerce')
    data[independent1] = pd.to_numeric(data[independent1], errors='coerce')
    
    # If only one independent variable is provided, perform a one-way ANOVA
    groups = data.groupby(independent1)[dependent].apply(list)
    f_statistic, p_value = scipy_stats.f_oneway(*groups)

    print(f"F-statistic: {f_statistic}")
    print(f"P-value: {p_value}")

    # Interpret the results
    if p_value < 0.05:
        print(f"There is a significant difference between the groups for {independent1}.")
    else:
        print(f"There is no significant difference between the groups for {independent1}.")

def perform_logistic_regression(file_path, dependent='key-eval', independent='Avg Key Var'):
    # Load the data
    print(f"Testing {file_path}")
    data = pd.read_csv(file_path)
    
    # Convert variables to numeric, coercing errors to NaN (then drop NaNs)
    data[dependent] = pd.to_numeric(data[dependent], errors='coerce')
    data[independent] = pd.to_numeric(data[independent], errors='coerce')
    data = data.dropna(subset=[dependent, independent])
    
    # Check if the dependent variable is binary (0 or 1)
    unique_vals = data[dependent].unique()
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(f"The dependent variable '{dependent}' must be binary (0 or 1).")
    
    # Add a constant for the intercept term in logistic regression
    X = sm.add_constant(data[independent])
    y = data[dependent]
    
    # Fit the logistic regression model
    model = sm.Logit(y, X)
    result = model.fit()
    
    # Display results
    print(result.summary())

    # Interpretation of p-value for the independent variable
    p_value = result.pvalues[independent]
    print(p_value)
    if p_value < 0.05:
        print(f"There is a significant relationship between {independent} and {dependent}.")
    else:
        print(f"There is no significant relationship between {independent} and {dependent}.")
    
def perform_logistic_regression_with_interaction(file_path, dependent='Key-eval', ind_var1='Avg Key Var', ind_var2='Set Count'):
    # Load the data
    print(f"Testing {file_path}")
    data = pd.read_csv(file_path)
    
    # Convert variables to numeric, coercing errors to NaN (then drop NaNs)
    data[dependent] = pd.to_numeric(data[dependent], errors='coerce')
    data[ind_var1] = pd.to_numeric(data[ind_var1], errors='coerce')
    data[ind_var2] = pd.to_numeric(data[ind_var2], errors='coerce')
    data = data.dropna(subset=[dependent, ind_var1, ind_var2])
    
    # Check if the dependent variable is binary (0 or 1)
    unique_vals = data[dependent].unique()
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(f"The dependent variable '{dependent}' must be binary (0 or 1).")
    
    # Add interaction term
    data['Interaction'] = data[ind_var1] * data[ind_var2]
    
    # Add a constant for the intercept term in logistic regression
    X = sm.add_constant(data[[ind_var1, ind_var2, 'Interaction']])
    y = data[dependent]
    
    # Fit the logistic regression model
    model = sm.Logit(y, X)
    result = model.fit()
    
    # Display results
    print(result.summary())

    # Interpretation of p-values for each independent variable
    for var in [ind_var1, ind_var2, 'Interaction']:
        p_value = result.pvalues[var]
        print(f"P-value for {var}: {p_value}")
        if p_value < 0.05:
            print(f"There is a significant relationship between {var} and {dependent}.")
        else:
            print(f"There is no significant relationship between {var} and {dependent}.")

# Example usage
# perform_logistic_regression_with_interaction('your_file_path.csv')
    

def chitest(ns, corrects):
    # Check if the inputs are valid
    if len(ns) != len(corrects):
        print("The length of 'ns' and 'corrects' must be the same.")
        return
    
    # Collect groups based on provided datasets
    groups = []
    
    # Define group data as [correct, incorrect] counts
    for n, correct in zip(ns, corrects):
        if n is not None and correct is not None:
            groups.append([n * correct, n * (1 - correct)])
    
    # Convert groups list to a numpy array for the contingency table
    contingency_table = np.array(groups)
    
    # Perform the Chi-Square test if we have at least 2 groups
    if len(groups) > 1:
        try:
            chi2_stat, p_value, dof, expected = scipy_stats.chi2_contingency(contingency_table)
            print(f"chi2 = {chi2_stat}")
            print(f"p = {p_value}")
            print(f"Expected Frequencies: \n{expected}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("At least two datasets are required to perform the Chi-Square test.")


def perform_anova_multiple_datasets(dependent='Key-eval', *file_paths):
    # Initialize a list to hold the data for each file
    print(f"Testing ANOVA across {len(file_paths)} datasets.")
    group_data = []
    
    # Loop through each file path
    for file_path in file_paths:
        # Load the data
        data = pd.read_csv(file_path)
        
        # Ensure the dependent variable is numeric
        data[dependent] = pd.to_numeric(data[dependent], errors='coerce')
        
        # Drop NaN values in the dependent variable
        data = data.dropna(subset=[dependent])
        
        # Append the dependent variable values as a list to the group data
        group_data.append(data[dependent].tolist())
    
    # Perform the one-way ANOVA across the groups
    f_statistic, p_value = scipy_stats.f_oneway(*group_data)
    
    # Display the results
    print(f"F-statistic: {f_statistic}")
    print(f"P-value: {p_value}")
    
    # Interpret the results
    if p_value < 0.05:
        print(f"There is a significant difference between the datasets for {dependent}.")
    else:
        print(f"There is no significant difference between the datasets for {dependent}.")


def find_outliers_in_division(file_path, column_numerator, column_denominator):
    # Load data from the CSV file
    data = pd.read_csv(file_path)
    
    # Check if specified columns exist
    required_columns = {column_numerator, column_denominator, 'Iter', 'DateTime', 'Test_temp'}
    if not required_columns.issubset(data.columns):
        raise ValueError("One or more specified columns (including 'Iter', 'DateTime', 'Test_temp') do not exist in the data.")
    
    # Divide the values of the specified columns
    division_result = data[column_numerator] / data[column_denominator]
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) of the division result
    Q1 = division_result.quantile(0.25)
    Q3 = division_result.quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = division_result[(division_result < lower_bound) | (division_result > upper_bound)]
    
    # Print results with additional column data for each outlier
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    print("Outliers removed (Row number, division result, Iter, DateTime, Test_temp):")
    for row_index, value in outliers.items():
        iter_value = data.loc[row_index, 'Iter']
        datetime_value = data.loc[row_index, 'DateTime']
        test_temp_value = data.loc[row_index, 'Test_temp']
        print(f"Row {row_index + 2}: Division Result = {value}, Iter = {iter_value}, DateTime = {datetime_value}, Test_temp = {test_temp_value}")

    # Return only non-outliers
    non_outliers = division_result[(division_result >= lower_bound) & (division_result <= upper_bound)]
    return non_outliers



#----------------------------------------------------------------------------------

prompt1=f'''In the card game Set, three cards are a Set if for each property they have, all three cards are all the same or all different.
For example, a valid Set from these cards:\n\n
Blue, Ellipse, 1, Lines
Red, Squiggle, 3, Lines
Blue, Squiggle, 1, Empty
Green, Squiggle, 1, Filled
Red, Diamond, 1, Filled
Red, Ellipse, 2, Empty
Blue, Squiggle, 3, Lines
Green, Squiggle, 3, Lines
Red, Diamond, 1, Empty
Green, Ellipse, 2, Filled
Blue, Squiggle, 3, Empty
Blue, Diamond, 3, Empty\n
Would be:\n
Blue, Ellipse, 1, Lines
Green, Squiggle, 1, Filled
Red, Diamond, 1, Empty\n
Because the colors, shape and filling are all different, and the number is all equal.\n
Find a Set in the following 12 cards:\n'''
prompt2=f'''You are an expert at card games and logical reasoning. In the card game Set, three cards are a Set if for each property they have, all three cards are all the same or all different.
Find a Set in the following 12 cards:\n'''
prompt3=f'''I have a number of combinations of attributes, I'll call them tuples. I want you to find three tuples so that in the combination of Tuples, every attribute is either the same across each tuple, or completely different across each tuple.\n\n'''
prompt4=f'''In the card game Set, three cards are a Set if for each property they have, all three cards are all the same or all different. The following 12 cards make up a set position. List a valid set of three cards among these cards:'''

#GPT_variability(subject_prompt = prompt, test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.9, eval_temp=0.3, iter=100, dataset_name='Variability_GPT4_n100')
#GPT_classification(test_model='gpt-4o', iter=200, dataset_name='Classification_GPT4o_n200')
#GPT_default(subject_prompt = prompt, test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.9, eval_temp=0.3, iter=100, dataset_name='Sanitycheck_GPT4_n100_2')
#GPT_vision(10)
#GPT_Reasoning_Comparison(subject_prompt = prompt, test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.9, eval_temp=0.3, iter=50, dataset_name=f'Reasoning_comparison_GPT4_n50')

#GPT_default(subject_prompt = prompt4, test_model='gpt-4-turbo', eval_model='gpt-3.5-turbo', test_temp=0.1, eval_temp=0.3, iter=13, dataset_name='Default_GPT4_n13_temp1')
filename1='data_2024-09-12_20-29-52'
temp1 = f'Thesis code\\Data\\Default_GPT4_n100_temp1\\{filename1}.csv'
filename2='data_2024-09-12_19-08-40'
temp3 = f'Thesis code\\Data\\Default_GPT4_n100_temp3\\{filename2}.csv'
filename3='data_2024-09-12_18-16-33'
temp6 = f'Thesis code\\Data\\Default_GPT4_n100_temp6\\{filename3}.csv'
filename4='data_2024-05-28_10-25-04'
temp9 = f'Thesis code\\Data\\Default_GPT4_n100_temp9\\{filename4}.csv'
filename5='sum_temps'
pooled = f'Thesis code\\Data\\Default_GPT4_n400_alltemp\\{filename5}.csv'
filename6='data_2024-06-23_15-07-33'
abstract = f'Thesis code\\Data\\Abstract_GPT4_n100\\{filename6}.csv'
filename7='data_2024-06-20_15-00-30'
iad = f'Thesis code\\Data\\IAD_GPT4_n100\\{filename7}.csv'
filename8='ncards pooled'
ncards=f'Thesis code\\Data\\4-11 cards onetry\\{filename8}.csv'
filename9='data_2024-11-15_20-14-08'
smallgpt=f'Thesis code\\Data\\GPT-3.5_Selection_n100\\{filename9}.csv'


perform_logistic_regression(smallgpt, 'Key-eval', 'Avg Key Var')
perform_logistic_regression(smallgpt, 'Key-eval', 'Set Count')