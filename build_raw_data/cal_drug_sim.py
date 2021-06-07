from sklearn.metrics import jaccard_score
import numpy as np


def cal_jaccard_sim(filepath):
    drugs_fea = np.genfromtxt(filepath, delimiter=',')

    sim_matrix = np.zeros((drugs_fea.shape[0], drugs_fea.shape[0]))
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[0]):
            if i != j:
                sim_matrix[i][j] = jaccard_score(drugs_fea[i], drugs_fea[j])

    return sim_matrix


if __name__ == '__main__':
    drug_fea = np.genfromtxt('../data/drug_fea_enzyme.csv', delimiter=',')
    sim_target = cal_jaccard_sim('../data/drug_fea_enzyme.csv')
    np.savetxt('../data/drug_sim_enzyme.csv', sim_target, delimiter=',')
    print(np.sum(drug_fea[3]))
    print(np.sum(drug_fea[7]))
