#!/usr/bin/env python3

import csv
import random
import string
from enigma.machine import EnigmaMachine


ROTOR_CHOICES = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
REFLECTOR_CHOICES = ['B', 'C'] # Common reflectors
NUM_ROTORS = 3

def generate_random_settings():
    rotors = random.sample(ROTOR_CHOICES, NUM_ROTORS)
    reflector = random.choice(REFLECTOR_CHOICES)
    ring_settings = [random.randint(0, 25) for _ in range(NUM_ROTORS)]
    display_settings = ''.join(random.choices(string.ascii_uppercase, k=NUM_ROTORS))
    all_letters = list(string.ascii_uppercase)

    random.shuffle(all_letters)
    plugboard_letters = all_letters[:20] 
    plugboard_settings = ' '.join(
        [a + b for a, b in zip(plugboard_letters[::2], plugboard_letters[1::2])]
    )
    
    return {
        'rotors': ' '.join(rotors),
        'reflector': reflector,
        'ring_settings': ring_settings,
        'plugboard_settings': plugboard_settings,
        'display_settings': display_settings
    }

def randomize_scramble(original_text):
    settings = generate_random_settings()
    
    machine = EnigmaMachine.from_key_sheet(
        rotors=settings['rotors'],
        reflector=settings['reflector'],
        ring_settings=settings['ring_settings'], 
        plugboard_settings=settings['plugboard_settings']
    )
    
    machine.set_display(settings['display_settings'])
    
    encrypted_text = machine.process_text(original_text)

    return encrypted_text


if __name__ == "__main__":
    processed = []
    new_sonnet = 0
    sonnet = ''
    counter = 0

    paths = ['../Data/war.txt', '../Data/ulyses.txt', '../Data/bible.txt', '../Data/pg100.txt']
    for path in paths:
        for line in open(path, 'r'):
            line = line.strip()
            if len(line) < 10:
                continue

            counter += 1

            line = "".join(char for char in line if char.isalnum() or char.isspace())
            line = line.lower()

            if counter >= 10:
                if sonnet != '':
                    scrambled = randomize_scramble(sonnet)
                    processed.append((sonnet, scrambled))
                    new_sonnet = 0
                    sonnet = ''
                else:
                    new_sonnet = 0
                counter = 0
            
            sonnet += line + ' '

    random.shuffle(processed)

    with open('../Data/enigma_processed.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original', 'Scrambled'])
        for original, scrambled in processed:
            writer.writerow([original, scrambled])

    print("Processing complete. Data saved to enigma_processed.csv")