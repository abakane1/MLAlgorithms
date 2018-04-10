import numpy as np
import csv
import scipy.stats as stats

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


csvFile = open("testsD.csv", "r",)
reader = csv.reader(csvFile)

for item in reader:
    #print (item)
    if reader.line_num == 1:
        continue
    Vector_1 = np.array(item[2:], dtype=float)
    csvFile2 = open("fusionD.csv", "r")
    reader2 = csv.reader(csvFile2)
    for item2 in reader2:
        if reader2.line_num == 1:
            continue
        Vector_2 = np.array(item2[2:], dtype=float)
        #print (Vector_2)
        Cosvalue = cos_sim(Vector_1, Vector_2)
        Covvalue,Pvalue = stats.pearsonr(Vector_1,Vector_2)
        #with open("resultsD.csv","w") as csvSave:
        #    writer = csv.writer(csvSave)
        #    writer.writerow([item[0:2], item2[0:2],value])
        print(item[0:2], item2[0:2], Cosvalue, Covvalue, Pvalue)