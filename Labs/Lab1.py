import numpy as np

#### QUESTION 1
def list_to_array(myList):
    return np.array(myList)
# list1 = [1,2,3,4]
# convertedList = list_to_array(list1)
# print(type(convertedList))

#### QUESTION 2
def stdev(numbers):
    numbers = np.array(numbers)
    return numbers.std()
# list1 = [1,2,3,4]
# print(stdev(list1))

#### QUESTION 3
def percentile(numbers, perc):
    numbers = np.array(numbers)
    return np.percentile(numbers, perc*100)
# list1 = [1,2,3,4]
# print(percentile(list1, 0.5))

#### QUESTION 4
def sorting(numbers):
    numbers = np.array(numbers)
    return np.sort(numbers)
list1 = [10,3,7,2,4,1,20,2]
print(sorting(list1))