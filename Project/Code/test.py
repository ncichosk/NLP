#!/usr/bin/env python3
import os

file_path = '../Data/basic_processed_2.csv'

total_chars = 0
total_words = 0
lines = 0

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    with open(file_path, 'r') as file:
        for line in file:
            lines += 1
            
            first_entry = line.strip().split(',')[0]
            total_chars += len(first_entry)
            
            total_words += len(first_entry.split())

    if lines > 0:
        avg_chars = total_chars / lines
        avg_words = total_words / lines
    else:
        avg_chars = 0
        avg_words = 0

    print(f"Average characters (first column): {avg_chars:.2f}")
    print(f"Average words (first column):      {avg_words:.2f}")
    print(f"Total lines:                       {lines}")
    print(f"Total characters:                  {total_chars}")
    print(f"Total words:                       {total_words}")