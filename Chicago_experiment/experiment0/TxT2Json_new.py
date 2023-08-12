import json
import numpy as np

fp = open("5g_data_stress_cpu.txt",'r') #train1_5g_data.txt  hybrid_data

#variable_number = 24
variable_number = 32 + 48

lines = fp.readlines()
data = {}

for line in lines[1:]:
    line = line.strip('\n')
    ss = line.split(',')
    temp = [];
    #for i in range(variable_number):
    #    temp.append(ss[i+1])

    for i in range(variable_number):
        # if i != 3:
            temp.append(ss[i+1])
    
    data[ss[0]] = temp

#print(data)
with open('json_data.json', 'w') as outfile:
    json.dump(data, outfile)
    
