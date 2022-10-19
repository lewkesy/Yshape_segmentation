import torch.nn as nn
import numpy as np


def cosine_similarity(x1, x2):
    cos = nn.CosineSimilarity()
    return (1 - cos(x1, x2)).mean()


def data_process(fn, num=None):
    
    raw_data = np.genfromtxt(fn)
    data = {}
    for d in raw_data:
        if d[-1] not in data:
            data[d[-1]] = []
        
        data[d[-1]].append(d[:3])
    
    for label in data:
        data[label] = centralize_data(np.array(data[label]))
        if num is not None:
            idx = np.random.choice(data[label].shape[0], num)
            data[label] = data[label][idx]
    
    res_dict = {'name': fn}
    return data, res_dict


def centralize_data(data):
    if len(data.shape) == 3:
        data = data[0]

    offset = (np.max(data, axis=0) + np.min(data, axis=0)) / 2 - np.zeros(3,)
    data -= offset[None, :]
    data /= abs(data).max()
    data[:, [1, 2]] = data[:, [2, 1]]
    # data[:, 1] = data[:, 1] - data[:, 1].min() - 1

    return data


def loadply(fn):
    with open(fn, "r") as freader:
        header=True
        vertices_count=0
        primitives_count=0
        while header:
            line = freader.readline()
            str=line.strip().split(' ')
            if str[0]=='element':
                if str[1]=='vertex':
                    vertices_count=int(str[2])
                elif str[1]=='primitive':
                    primitives_count=int(str[2])
            elif str[0]=='end_header':
                header=False
            #otherwise continue
        pointset=[]

        for i in range(vertices_count):
            line = freader.readline()
            numbers=line.strip().split(' ')
            pt=[]
            pt.append(float(numbers[0]))
            pt.append(float(numbers[1]))
            pt.append(float(numbers[2]))
            pointset.append(pt)

        '''primitives=[]
        for i in range(primitives_count):
            line = freader.readline()
            numbers = line.strip().split(' ')
            pr=[]
            for j in range(len(numbers)):
                pr.append(float(numbers[j]))
            primitives.append(pr)'''

    return np.array(pointset)