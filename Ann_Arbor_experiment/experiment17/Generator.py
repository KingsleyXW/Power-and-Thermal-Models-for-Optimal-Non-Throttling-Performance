import random
import string

with open('data.txt', 'w') as f:
    for i in range(0,100000000):
        str = ''.join(random.choices(string.ascii_lowercase, k=5))
        for ii in range(0, 10):
            f.write(str)
    f.close()
