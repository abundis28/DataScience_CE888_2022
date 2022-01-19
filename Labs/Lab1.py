import numpy as np

#### QUESTION 1
def list_to_array(myList):
    ## Your code goes here
    return np.array(myList)
# list1 = [1,2,3,4]
# convertedList = list_to_array(list1)
# print(type(convertedList))

#### QUESTION 2
def stdev(numbers):
    ## Your code goes here
    numbers = np.array(numbers)
    return numbers.std()
# list1 = [1,2,3,4]
# print(stdev(list1))
