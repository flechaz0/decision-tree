import tree as tree
import numpy as np
def Preprocess(filename):
    data_file = open(filename,'r')
    data_set = []
    for line in data_file.readlines():
        t_array = line.strip('.\n').split(', ')
        if '?' in t_array:
            continue
        temp_array1 = np.array(t_array)
        temp_array2 = np.delete(temp_array1, 13)
        t = np.delete(temp_array2, 2)
        data_set.append(t)
    return data_set

mytree = tree.grabTree('abc.pkl')
test_file = 'adult.test'
labels = ['Age','Workclass','Education','EdNum','MaritalStatus',
           'Occupation','Relationship','Race','Sex','CapitalGain',
           'CapitalLoss','HoursPerWeek']

test_data = Preprocess(test_file)
num_accuracy = 0


for vec in test_data:
    predict_class = tree.classify(mytree,labels,vec)
    if predict_class == vec[-1]:
        num_accuracy += 1
accuracy = num_accuracy / len(test_data)
print(accuracy)





