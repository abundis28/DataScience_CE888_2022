import numpy as np

## Question 1
def list_to_array(myList):
    ## Your code goes here
    return np.array(myList)

list1 = [1,2,3,4]

convertedList = list_to_array(list1)

print(type(convertedList))