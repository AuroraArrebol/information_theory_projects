"""
Author：ZWP
U202112277
"""
import numpy as np
import matplotlib.pyplot as plt

def load_txt(file_path):
    '''
    Load txt document, pre-process text
    :param file_path: File path
    :return:  a very long string
    '''
    with open(file_path, encoding='utf8', errors='ignore') as f:
        lines = f.readlines()
    lines = [line.strip().lower() for line in lines]
    total = lines[0]
    for line in lines:
        total += line
    return total


def calc_entropy_str(string):
    """
    Counting huge strings and calculating entropy
    :param string: huge strings
    :return: entropy,word probability
    """
    dictionary = {}
    lenth = len(string)
    for i in string:
        if i in dictionary.keys():
            dictionary[i] += 1 / len(string)
        else:
            dictionary[i] = 1 / len(string)
    array = np.array([float(i) for i in dictionary.values()])
    sum = 0
    for i in array:
        sum += -i * np.log2(i)
    #Return the Information entropy and the word probability from highest to lowest
    return sum, sorted(dictionary.items(), key=lambda x: -x[1])


def cac_entropy_rate(dist,mat):
    '''
    Calculate the entropy rate for the data in the list
    :param dist: stable distribution
    :param mat: Transition Matrix
    :return: entropy rate
    '''
    mat_=mat
    mat_=mat_+(mat_==0)*1
    ans=-dist*mat*np.log2(mat_)
    return np.sum(ans)


def get_Markov_Transition_Matrix(string, order):
    '''
    Generate the Markov Transition Matrix based on the
    order of the Markov chain and the string
    :param string:the long string
    :param order:the order of markov chain
    :return:
    '''
    lenth = len(string)

    if order == 0:
        #Each row of the transition matrix is the same when the order is 0
        dictionary = {}
        for i in string:
            if i in dictionary.keys():
                dictionary[i] += 1
            else:
                dictionary[i] = 1
        dict_len = len(dictionary)
        distribution = np.array([float(i) for i in dictionary.values()]) / lenth
        matrix = np.vstack([distribution for _ in range(dict_len)])
        return matrix
    else:
        # List of strings according to the n-gram, each element of the list has a lenth of n
        str_list = []
        for i in range(lenth - order + 1):
            element = string[i]
            for j in range(1, order):
                element += string[i + j]
            str_list.append(element)
        dictionary = {}
        for i in str_list:
            if i not in dictionary.keys():
                dictionary[i] = len(dictionary)
        matrix = np.zeros((len(dictionary), len(dictionary)))
        for i in range(len(str_list) - 1):
            matrix[dictionary[str_list[i]], dictionary[str_list[i + 1]]] += 1
        qie = np.sum(matrix, axis=1, keepdims=True)
        qie = qie + (qie == 0) * 1
        matrix = matrix / qie
        return matrix


def get_stalble_prob(matrix):
    '''
    Obtain stable probability distribution based on probability transition matrix
    :param matrix: transition matrix
    :return:stable probability distribution
    '''
    stalble_prob = np.ones((1, matrix.shape[0])) / matrix.shape[0]
    while (1):
        stalble_prob_ = stalble_prob @ matrix
        '''
        When the difference between the results of the two operations 
        has a second-order norm less than 10^-3, the operation ends
        '''
        if np.linalg.norm(stalble_prob_ - stalble_prob) < 1e-3:
            break
        stalble_prob = stalble_prob_

    return stalble_prob

if __name__=='__main__':
    string_en = load_txt("English.txt")
    string_cn = load_txt("Chinese.txt")
    entropy_en, distribution_en = calc_entropy_str(string_en)
    entropy_cn, distribution_cn = calc_entropy_str(string_cn)
    print("The entropy of the English speech:", entropy_en)
    print("The entropy of the Chinese speech:", entropy_cn)


    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 13

    plt.figure(num=1, figsize=(18, 10))
    plt.title("Distibution of English Speech (highest 20)", size=26)
    plt.bar([("space" if (j == 0) else i[0]) for j, i in enumerate(distribution_en) if j < 20],
            [i[1] for j, i in enumerate(distribution_en) if j < 20],
            width=0.5, bottom=0, align='center', color='g', edgecolor='r', linewidth=2)
    plt.xlabel("letters and punctuation", size=28)
    plt.ylabel("frequncy", size=28)

    plt.figure(num=2, figsize=(18, 10))
    plt.title("Distibution of Chinese Speech (highest 20)", size=26)
    plt.bar([i[0] for j, i in enumerate(distribution_cn) if j < 20],
            [i[1] for j, i in enumerate(distribution_cn) if j < 20],
            width=0.5, bottom=0, align='center', color='g', edgecolor='r', linewidth=2)
    plt.xlabel("letters and punctuation", size=28)
    plt.ylabel("frequncy", size=28)

    fig=plt.figure(figsize=(18, 10))
    for i, string in enumerate([string_en, string_cn]):
        for j,order in enumerate([0, 3, 5]):
            markov_mat = get_Markov_Transition_Matrix(string, order)
            stalble_prob = get_stalble_prob(markov_mat)
            entropy_rate = cac_entropy_rate(stalble_prob,markov_mat)
            if (i == 0):
                language = "English"
            else:
                language = "Chinese"
            plt.subplot(2, 3, 3 * i + (j + 1))
            plt.title("Markov Transition Matrix in "+language+" (order:"+ str(order)+")", size=10)
            plt.imshow(markov_mat,cmap="Reds")
            print("The " + language + " speech's entropy rate is (order:" + str(order) + "):", entropy_rate)
    plt.show()