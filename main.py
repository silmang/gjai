import myMath

print("pi value: ", myMath.pi)
sum1 = myMath.sum1toN(10)
print(sum1)

mu1 = myMath.multiply1toN(10)
print(mu1)

from myMath import sum1toN

print(sum1toN(10))

import os

print(os.getcwd())

import myMath as mm

print(mm.sum1toN(10))

import keyword

print(keyword.kwlist)

import random

for i in range(6):
    number = random.randint(1, 45)
    print(number, end=' ')
print()

num_list = sorted(random.sample(range(1, 45), 6))
print(num_list)

print(random.random())
print(random.uniform(0, 10))

import numpy as np

print(np.random.random(3))
print(np.random.uniform(0, 10, 3))

import time

def manyloop(num):
    start = time.time()
    print(f"start: {start}")

    for i in range(num):
        if i%10000==0:
            print("*", end="")
    print()

    end = time.time()
    print(f"end: {end}")
    print(end - start, '초 경과')

# number = int(input("몇 번을 반복할까요?: "))
# manyloop(number)

from tqdm import tqdm

current = time.ctime()
print(current)

current_list = current.split()
print(type(current_list))

for t in tqdm(current_list):
    print(t)

for t in range(6):
    print(time.ctime())
    time.sleep((1))












import random
import time

time.sleep(1)
num = random.randint(1, range(101)[-1])
print(num)




