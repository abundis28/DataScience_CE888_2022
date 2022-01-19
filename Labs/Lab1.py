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
# list1 = [10,3,7,2,4,1,20,2]
# print(sorting(list1))

#### QUESTION 5
def mean_of_array_rows(numbers):
    numbers = np.array(numbers)
    return np.mean(numbers,axis=1)
# list1 = [[1,3,5],[2,4,6],[11,13,9]]
# print(mean_of_array_rows(list1))

#### QUESTION 6
def percentile_array_columns(myArray, perc):
    return np.percentile(myArray, perc, axis=0)
# list1 = np.asarray([55, 88, 78, 90, 79, 94]).reshape(2, -1)
# print(percentile_array_columns(list1, 60))

#### QUESTION 7
# Define our actions
def action_0():
    ''' This should return 1 with probability 50%, and 0 with probability 50%'''
    return np.random.choice(a=[1,0],p=[0.5, 0.5])

def action_1():
    ''' This should return 1 with probability 60%, and 0 with probability 40%'''
    return np.random.choice(a=[1,0],p=[0.6, 0.4])

def action_2():
    ''' This should return 1 with probability 20%, and 0 with probability 80%'''
    return np.random.choice(a=[1,0],p=[0.2,0.8])

# np.random.seed(4)
# print('%d, %d, %d, %d, %d, %d' % (action_0(), action_1(), action_2(), action_0(), action_1(), action_2()))
