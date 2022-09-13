def sum1toN(num):
    return num*(num+1)/2
def multiply1toN(num):
    result = 1
    for i in range(1, num+1):
        result *= i
    return result
pi = 3.1415926535