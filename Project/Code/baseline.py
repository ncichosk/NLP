#!/usr/bin/env python3

import csv
from enigma.machine import EnigmaMachine

text = []
scrambles = []
firstline = 1
line_count = 0

letter_freq = {
    "E": 12.02,
    "T": 9.10,
    "A": 8.12,
    "O": 7.68,
    "I": 7.31,
    "N": 6.95,
    "S": 6.28,
    "R": 6.02,
    "H": 5.92,
    "D": 4.32,
    "L": 3.98,
    "U": 2.88,
    "C": 2.71,
    "M": 2.61,
    "F": 2.30,
    "Y": 2.11,
    "W": 2.09,
    "G": 2.03,
    "P": 1.82,
    "B": 1.49,
    "V": 1.11,
    "K": 0.69,
    "X": 0.17,
    "Q": 0.11,
    "J": 0.10,
    "Z": 0.07
}

######################################################################################
# Guess on the enigma approach
######################################################################################

with open('../Data/nearby-indicators.csv', newline='') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for line in lines:
        if firstline:
            firstline = 0
            continue
        text.append(line[3])
        scrambles.append(line[2])
        line_count += 1

total_correct = 0
total = 0

for line in range(2):
    freq_dict = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "H": 0, "I": 0, "J": 0, "K": 0, "L": 0, "M": 0, "N": 0, "O": 0, "P": 0, "Q": 0, "R": 0, "S": 0, "T": 0, "U": 0, "V": 0, "W": 0, "X": 0, "Y": 0, "Z": 0}
    for char in scrambles[line]:
        if char in freq_dict:
            freq_dict[char] += 1

    freq_list = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    freq_order = [item[0] for item in freq_list]
    real_freq_order = sorted(letter_freq.items(), key=lambda x: x[1], reverse=True)
    real_freq_order = [item[0] for item in real_freq_order]

    line_chars = list(scrambles[line])
    for i in range(len(line_chars)):
        line_chars[i] = real_freq_order[freq_order.index(line_chars[i])]
    scrambles[line] = ''.join(line_chars)

    for t_char, s_char in zip(text[line], scrambles[line]):
        if t_char == s_char:
            total_correct += 1
        total += 1


print("Accuracy on Enigma: " + str(total_correct / total))

######################################################################################
# Now for Scramble approach
######################################################################################

with open('../Data/basic_processed.csv', newline='') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for line in lines:
        if firstline:
            firstline = 0
            continue
        text.append(line[0])
        scrambles.append(line[1])
        line_count += 1

total_correct = 0
total = 0

for line in range(2):
    freq_dict = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "H": 0, "I": 0, "J": 0, "K": 0, "L": 0, "M": 0, "N": 0, "O": 0, "P": 0, "Q": 0, "R": 0, "S": 0, "T": 0, "U": 0, "V": 0, "W": 0, "X": 0, "Y": 0, "Z": 0}
    for char in scrambles[line]:
        if char in freq_dict:
            freq_dict[char] += 1

    freq_list = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    freq_order = [item[0] for item in freq_list]
    real_freq_order = sorted(letter_freq.items(), key=lambda x: x[1], reverse=True)
    real_freq_order = [item[0] for item in real_freq_order]

    line_chars = list(scrambles[line])
    for i in range(len(line_chars)):
        line_chars[i] = real_freq_order[freq_order.index(line_chars[i])]
    scrambles[line] = ''.join(line_chars)

    for t_char, s_char in zip(text[line], scrambles[line]):
        if t_char == s_char:
            total_correct += 1
        total += 1


print("Accuracy on Scramble: " + str(total_correct / total))