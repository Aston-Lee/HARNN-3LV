import os
import json
import statistics 
import time
import numpy as np
from sklearn.metrics import accuracy_score
threadhold = [0,5,26,246,1000 ]

def get_predition(file_name):
    acc_list = []
    '''
    hira_label_list = np.array( [[],[],[],[]])
    hira_predi_list = np.array( [[],[],[],[]])
    '''
    hira_label_list = [[],[],[],[]]
    hira_predi_list = [[],[],[],[]]
    file_name = os.path.join('output' , file_name,'predictions.json')
    print(file_name)
    with open(file_name , 'r') as handle:
        json_data = [json.loads(line) for line in handle]

    
    '''
    for data in json_data:
        label = data['labels']
        predi = data['predict_labels']
        acc_list.append(subset_acc(label,predi))
        print(acc_list)
        subset_accuracy = statistics.mean(acc_list)
    print(subset_accuracy)
    '''

    for data in json_data:
        label = data['labels']
        predi = data['predict_labels']


        for thred in range(0,4):
            hira_label = depart_label(label,thred)
            hira_predi = depart_label(predi,thred)
            lab,pre = return_onehot(hira_label,hira_predi,thred) 
            hira_label_list[thred].append(lab)
            hira_predi_list[thred].append(pre)


            #hira_acc_list[thred].append(subset_acc(hira_label,hira_predi,thred))
            #acc_list.append(subset_acc(hira_label,hira_predi))

    b = np.array(hira_label_list[1])
    a = np.array(hira_predi_list[1])
    print(accuracy_score(a,b))
    return
    print(hira_predi_list[0])
    for thred in range(0,4):
        subset_acc = accuracy_score(hira_label_list[thred], hira_predi_list[thred])
        print(subset_acc)


def depart_label(label,thred):
    hira_label = []
    for i in label:
        if i < threadhold[thred + 1] and i > threadhold[thred]:
            hira_label.append(i)
    return hira_label




def return_onehot(label,predi,thred):
    thred = threadhold[thred + 1] - threadhold[thred] - 1 
    label = one_hot(label,thred)
    predi = one_hot(predi,thred)

    #subset_acc = accuracy_score(label, predi)
    return label ,predi
    #return subset_acc
    
def one_hot(alist,thred):
    one_hot_list = []
    for i in range(thred):
        if i in alist:
            one_hot_list.append(1)
        else:
            one_hot_list.append(0)
    return one_hot_list



get_predition('1593586088')


