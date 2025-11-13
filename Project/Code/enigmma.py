#!/usr/bin/env python3

import csv
from enigma.machine import EnigmaMachine

text = ["fromfairestcreatureswedesireincreasethattherebybeautysrosemightneverdiebutastheripershouldbytimedeceasehistenderheirmightbearhismemorybutthoucontractedtothineownbrighteyesfeedstthylightsflamewithselfsubstantialfuelmakingafaminewhereabundanceliesthyselfthyfoetothysweetselftoocruelthouthatartnowtheworldsfreshornamentandonlyheraldtothegaudyspringwithinthineownbudburiestthycontentandtenderchurlmakstwasteinniggardingpitytheworldorelsethisgluttonbetoeattheworldsduebythegraveandthee"
]
scrambles = []
firstline = 1
line_count = 0

'''
with open('../Data/nearby-indicators.csv', newline='') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for line in lines:
        if firstline:
            firstline = 0
            continue
        text.append(line[3])
        scrambles.append(line[2])
        line_count += 1
'''
total_correct = 0
total = 0

machine = EnigmaMachine.from_key_sheet(
    rotors='VII IV V',
    reflector='B',
    ring_settings=[23, 10, 6], 
    plugboard_settings='AC LS BQ WN MY UV FJ PZ TR OK'
)

machine.set_display('WJF')

'''
for line in range(1):
    decrypted_text = machine.process_text(scrambles[line])

    for t_char, s_char in zip(text[line], decrypted_text):
        if t_char == s_char:
            total_correct += 1
        total += 1

print("Accuracy: " + str(total_correct / total))
'''

encrypted_text = machine.process_text(text[0])
print("Encrypted Text: " + encrypted_text)