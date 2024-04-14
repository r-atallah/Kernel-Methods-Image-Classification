import argparse
import pandas as pd
import numpy as np
from utils import *
from hog import HOGFeatureExtractor
from fisher_vector import FisherVectorExtractor
from kernel_pca import KernelPCA
from kernels import GaussianKernel,LinearKernel,LaplacianRBFKernel,HellingerKernel,SublinearRBFKernel,GaussianKernel_orientation
from KRC import KernelRidgeClassifier
from SVM import OneVsAllClassifier,SVM_perso,SVM

def parse_args():
    parser = argparse.ArgumentParser(prog='start',
                                     description="Kernel methods for Machine Learning")

    parser.add_argument('--input_path', default='./data',type=str, help="PATH OF TRAIN/TEST SAMPLES/LABELS")
    parser.add_argument('--output_path', type=str, default='./data', help="PATH TO STORE THE FINAL OUTPUT")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(42)
    # DATA LOADING
    Xtr,Xte,Ytr = read_data(args.input_path)
    print(f"data loaded:\n\t{len(Xtr)} training images\n\t{len(Ytr)} training labels\n\t{len(Xte)} test images")

    # TRAIN DATA AUGMENTATION 
    Xtr, Ytr = augment_data(Xtr, Ytr, flip_ratio=1, rot_ratio=1, rot_replicas=1, rot_angle=30)
    print(f"after data augmentation:\n\t{len(Xtr)} training images\n\t{len(Ytr)} training labels")
    X_train, X_test= reshape(Xtr) , reshape(Xte)

    # HOG FEATURES + PCA
    print('Extracting HOG')
    hog = HOGFeatureExtractor()
    X_train_hog, X_test_hog = hog.transform(X_train), hog.transform(X_test)
    pca = KernelPCA(LinearKernel(),r=85)
    pca.fit(X_train_hog)
    X_train_hog_pca = pca.predict(X_train_hog)
    X_test_hog_pca = pca.predict(X_test_hog)

    # HOG FEATURES + FISHER VECTOR + PCA
    print('Extracting HOG FISHER')
    hog = HOGFeatureExtractor()
    X_train, X_test = hog.transform(X_train,unflatten=True), hog.transform(X_test,unflatten=True)
    hog_fisher = FisherVectorExtractor(nclasses=256)
    X_train_hog_fisher = hog_fisher.train(X_train)
    X_test_hog_fisher = hog_fisher.predict(X_test)
    pca = KernelPCA(LinearKernel(),r=210)
    pca.fit(X_train_hog_fisher)
    X_train_hog_fisher_pca = pca.predict(X_train_hog_fisher)
    X_test_hog_fisher_pca = pca.predict(X_test_hog_fisher)
    X_train_hog_fisher_pca.shape

    # TRAINING
    print('TRAINING ...')
    krc = KernelRidgeClassifier(kernel='rbf',gamma=1.2,C=1)
    svm_classifier = OneVsAllClassifier(SVM,num_classes=len(np.unique(Ytr)),c=10, kkt_thr=2e-3,max_iter=4e3,kernel_type='rbf',gamma_rbf=0.23)
    svm_classifier2 = OneVsAllClassifier(SVM,num_classes=len(np.unique(Ytr)),c=100, kkt_thr=2e-3,max_iter=4e3,kernel_type='rbf',gamma_rbf=1.5)
    krc.fit(X_train_hog_pca, Ytr)
    svm_classifier.fit(X_train_hog_pca, Ytr)
    svm_classifier2.fit(X_train_hog_fisher_pca, Ytr)

    # PREDICTION
    print('TESTING ...')
    labels_krc = krc.predict(X_test_hog_pca).reshape(-1)
    labels_svm = svm_classifier.predict(X_test_hog_pca)
    labels_svm2 = svm_classifier2.predict(X_test_hog_fisher_pca)
    labels=[]
    for i in range(len(labels_krc)):
        predictions = [labels_krc[i], labels_svm[i], labels_svm2[i]]
        counts = {label: predictions.count(label) for label in set(predictions)}
        max_count = max(counts.values())
        if list(counts.values()).count(max_count) == 1:
            majority_vote = max(counts, key=counts.get)
        else:
            majority_vote = labels_svm[i]  # Default to labels_svm in case of tie
        labels.append(majority_vote)
        
    # EXPORT
    print('EXPORTING ...')
    Yte = {'Prediction' : labels} 
    dataframe = pd.DataFrame(Yte) 
    dataframe.index += 1 
    dataframe.to_csv(f'{args.output_path}/Yte_pred.csv',index_label='Id')
