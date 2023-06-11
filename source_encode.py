"""
姓名:钟伟鹏
ID:U202112277
"""
import numpy as np
import math
import copy

class Node:
    """
    The node in Huffman code tree
    """
    def __init__(self,val:int,letter:str=None,child:list=None):
        self.val=val
        self.letter = letter
        self.child = child
        self.code = []

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
            dictionary[i] += 1 /lenth
        else:
            dictionary[i] = 1 /lenth
    array = np.array([float(i) for i in dictionary.values()])
    sum = 0
    for i in array:
        sum += -i * np.log2(i)
    #Return the Information entropy and the word probability from highest to lowest
    return sum, {k: v for k, v in sorted(dictionary.items(), key=lambda x: x[1])}

def Huffman_code_tree(word_dict,Q):
    """
    返回Huffman树的根节点
    :param word_dict: 字典的键值对是 {字符：字符出现次数}
    :param Q: Huffman树是Q叉树
    :return: Huffman树的根节点
    """
    length=len(word_dict)
    assert Q>=2
    #增加节点使得可以得到好的Hudffman树
    if(Q>2):
        add_node_num=Q-1-(length-Q)%(Q-1)
        for i in range(add_node_num):
            word_dict["add"+str(i+1)]=0
    #升序排列
    word_dict={k: v for k, v in sorted(word_dict.items(), key=lambda x: x[1])}
    #为每一个值建立一个树节点
    Nodes=[Node(v,k) for k,v in word_dict.items()]

    #建立Huffman树
    while len(Nodes)>1:
        child_nodes=[node for i,node in enumerate(Nodes) if i<Q]    #child_nodes是将要合并的子节点
        sum=int(np.array([node.val for node in child_nodes]).sum()) #对子节点的val求和
        new_node=Node(sum,child=child_nodes)                        #新建节点
        Nodes.append(new_node)                                      #加入新节点
        Nodes=Nodes[Q:]                                             #删除合并的节点
        Nodes=sorted(Nodes,key=lambda x:x.val)                      #重新升序排列
    root=Nodes[0]
    return root                                                     #返回根节点

def get_Huffman_code(root,code_dict):
    if root.child==None:
        code_dict[root.letter]=root.code
    else:
        for i,node in enumerate(root.child):
            for code in root.code:
                node.code.append(code)
            node.code.append(i)
            get_Huffman_code(node,code_dict)

def Shannon_code(word_dict,Q):
    #升序排列
    word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda x: -x[1])}
    # print(word_dict)
    code_dict={}
    Pa=[0]
    for letter,p in word_dict.items():
        code_length=int(math.log(1/p,Q))+1
        code=get_code(Pa[-1],code_length,Q)
        code_dict[letter]=code
        Pa.append(Pa[-1]+p)
    return code_dict

def get_code(pa,lenth,Q):
    code=[]
    for i in range(lenth):
        pa*=Q
        code.append(int(pa>=1))
        pa= pa if pa<1 else pa-1
    return code

if __name__=='__main__':
    #entropy
    string = load_txt("Jobs_speech.txt")
    entropy, distribution_dict = calc_entropy_str(string)
    print("The entropy of the speech:", entropy)

    #input Q
    Q=int(input("Please input the code alphabet length Q:"))
    print(distribution_dict)
    #Huffman encode
    root=Huffman_code_tree(copy.deepcopy(distribution_dict), Q)
    code_dict={}
    get_Huffman_code(root,code_dict)
    code_dict= {k:v for k,v in code_dict.items() if len(k)==1}
    print([len(i) for i in code_dict.values()])
    average_code_length = np.array([len(code_dict[key]) * distribution_dict[key] for key in code_dict.keys()]).sum()
    print("the average Huffman code length is:",average_code_length)
    print("the Huffman code(sorted by code length) is: ", code_dict)
    code_dict= {k:v for k,v in sorted(code_dict.items(), key=lambda x: x[0])}
    print("the Huffman code(sorted by string) is: ",code_dict)
    print("\n")
    #Shannon_encode
    code_dict=Shannon_code(distribution_dict,Q)
    code_dict = {k: v for k, v in code_dict.items() if len(k) == 1}
    print([len(i) for i in code_dict.values()])
    average_code_length = np.array([len(code_dict[key]) * distribution_dict[key] for key in code_dict.keys()]).sum()
    print("the average Shannon code length is:", average_code_length)
    print("the Shannon code(sorted by code length) is: ", code_dict)
    code_dict = {k: v for k, v in sorted(code_dict.items(), key=lambda x: x[0])}
    print("the Shannon code(sorted by string) is: ", code_dict)



