#!/usr/bin/env python3

import csv
import random
import string

def scramble(sonnet):
    alphabet = list(string.ascii_lowercase)
    shuffled_alphabet = list(string.ascii_lowercase)
    random.shuffle(shuffled_alphabet)
    
    lower_map = dict(zip(alphabet, shuffled_alphabet))
    
    scrambled_text = []
    for char in sonnet:
        scrambled_text.append(lower_map.get(char, char))
        
    return "".join(scrambled_text)



if __name__ == "__main__":
    processed = []
    new_sonnet = 0
    sonnet = ''

    for line in open('../Data/basic.txt'):
        line = line.strip()
        if line.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')) or line == '':
            new_sonnet = 1
            continue

        line = "".join(char for char in line if char.isalnum())
        line = line.lower()

        if new_sonnet:
            if sonnet != '':
                scrambled = scramble(sonnet)
                processed.append((sonnet, scrambled))
                new_sonnet = 0
                sonnet = ''
            else:
                new_sonnet = 0
        
        sonnet += line

    with open('../Data/basic_processed.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original', 'Scrambled'])
        for original, scrambled in processed:
            writer.writerow([original, scrambled])

    print("Processing complete. Data saved to basic_processed.csv")