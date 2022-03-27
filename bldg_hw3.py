import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import load_svmlight_file
import utils
from scipy.linalg import pinv
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA 
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.mixture import GaussianMixture
from sklearn.metrics import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


#setup the randoms tate
RANDOM_STATE = 19920604


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
	#NOTE: It is important to provide the output in the same order
	return accuracy_score(Y_true, Y_pred)

#input: Name of classifier, predicted labels, actual labels
#output: print ACC, AUC, Prec, Recall and F1-Score of the Classifier
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print("______________________________________________")
	print("")

def main():
    #load training and testing data
    X = pd.read_csv("bldg_x.csv")
    Y = pd.read_csv("bldg_y.csv")
    X_train = X[:70]
    X_test = X[71:]
    Y_train = Y[:70]
    Y_test = Y[71:]
    Y_train = Y_train.values.ravel()
    Y_test = Y_test.values.ravel()

    #Kmeans
    #change and select params here
    range_n_clusters = np.arange(2,30,1)
    inertia = []
    homogeneity = []
    silhouette = []
    for n_clusters in range_n_clusters:
        k_means_clustering = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        k_means_clustering.fit(X_train)
        inertia.append(k_means_clustering.inertia_)
        homogeneity.append(homogeneity_score(Y_train, k_means_clustering.labels_))
        silhouette.append(silhouette_score(X_train, k_means_clustering.labels_, metric="euclidean"))

    #plot inertia for Kmeans
    inertia = np.array(inertia)    
    plt.figure()
    plt.plot(range_n_clusters,inertia)
    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    plt.title('Clusters vs Inertia for bldg')
    plt.grid()
    plt.savefig('bldg_kmeans_inertia.png')
    
    #plot homogeneity score for Kmeans
    homogeneity = np.array(homogeneity)    
    plt.figure()
    plt.plot(range_n_clusters,homogeneity)
    plt.xlabel('Clusters')
    plt.ylabel('Homogeneity')
    plt.title('Clusters vs Homogeneity for bldg')
    plt.grid()
    plt.savefig('bldg_kmeans_homogeneity.png')
    
    #plot silhouette score for Kmeans
    silhouette = np.array(silhouette)    
    plt.figure()
    plt.plot(range_n_clusters,silhouette)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette')
    plt.title('Clusters vs Silhouette for bldg')
    plt.grid()
    plt.savefig('bldg_kmeans_silhouette.png')
    
    #GMM (EM)
    #lowest_bic = np.infty
    n_components_range = range(1, 30)
    covariances = ['spherical', 'tied', 'diag', 'full']
    bic = np.zeros((len(covariances),len(n_components_range)))
    for i, covariance in enumerate(covariances):
        for j, n_components in enumerate(n_components_range):
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance)
            gmm.fit(X_train)
            bic[i][j] = gmm.bic(X_train)
            # if bic[i][j] < lowest_bic:
            #     lowest_bic = bic[i][j]
            #     best_gmm = gmm
    #plot BIC for EM
    plt.figure()
    plt.plot(n_components_range, bic[0], label = 'Spherical')
    plt.plot(n_components_range, bic[1], label = 'Tied')
    plt.plot(n_components_range, bic[2], label = 'Diag')
    plt.plot(n_components_range, bic[3], label = 'Full')
    plt.legend()
    plt.xticks(np.arange(0, len(n_components_range)+1, 4.0))
    plt.title("Number of Components vs. BIC for Bldg")
    plt.xlabel("Number of Components")
    plt.ylabel("BIC")
    plt.savefig('bldg_gmm_bic.png')
    plt.show()
    
    #PCA standardize
    X_train_PCA = StandardScaler().fit_transform(X_train)
    #plot PCA 2D
    pca_2D = PCA(n_components=2)
    pca_2D.fit(X_train_PCA)
    print('2D ratios',pca_2D.explained_variance_ratio_)
    print('2D eigenvalues',pca_2D.explained_variance_)
    x_2D = pca_2D.fit_transform(X_train_PCA)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PCA 1st Component')
    ax.set_ylabel('PCA 2nd Component')
    ax.set_title('Bldg Data Reduced to 2D by PCA')
    for i in range(len(Y_train)):
        if Y_train[i] == 0:
            ax.scatter(x_2D[i, :][0], x_2D[i, :][1], c = 'b', marker='o', label='0')
        elif Y_train[i] == 1:
            ax.scatter(x_2D[i, :][0], x_2D[i, :][1], c = 'r', marker='o', label='1')
    ax.grid()
    plt.title('Bldg Data Reduced to 2D by PCA')
    plt.savefig('bldg_pca_2d.png')
    plt.show()
    
    #plot PCA 3D
    pca_3D = PCA(n_components = 3)
    pca_3D.fit(X_train_PCA)
    print('3D ratios',pca_3D.explained_variance_ratio_)
    print('3D eigenvalues',pca_3D.explained_variance_)
    x_3D = pca_3D.fit_transform(X_train_PCA)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(Y_train)):
        if Y_train[i] == 0:
            ax.scatter(x_3D[i, :][0], x_3D[i, :][1], x_3D[i, :][2], c = 'b', marker='o', label='0')
        elif Y_train[i] == 1:
            ax.scatter(x_3D[i, :][0], x_3D[i, :][1], x_3D[i, :][2], c = 'r', marker='o', label='1')
    ax.set_xlabel('PCA 1st Component')
    ax.set_ylabel('PCA 2nd Component')
    ax.set_zlabel('PCA 3rd Component')
    plt.title('Bldg Data Reduced to 3D by PCA')
    plt.savefig('bldg_pca_3d.png')
    plt.show()
    
    #plot PCA variance/eigenvalues
    pca = PCA(n_components=13)
    pca.fit(X_train_PCA)
    plt.figure()
    plt.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), pca.explained_variance_ratio_, label='var')
    plt.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), np.cumsum(pca.explained_variance_ratio_), label='cum var')
    plt.xticks(np.arange(1, pca.explained_variance_ratio_.size + 1, 2))
    plt.xlabel('Component')
    plt.ylabel('Variance')
    plt.title('Cumulative Variance and Variance Distribution for Bldg')
    plt.legend()
    plt.grid()
    plt.savefig('bldg_pca_variance.png')
    #best PCA X
    X_PCA_best = PCA(n_components = 9).fit_transform(X_train_PCA)
    
    #ICA
    #ICA plot kurtosis
    X_train_ICA = StandardScaler().fit_transform(X_train)
    kurtosis_values = []
    for i in range(1,30):
        X_ICA = FastICA(n_components = i).fit_transform(X_train_ICA)
        kur = scipy.stats.kurtosis(X_ICA)
        kurtosis_values.append(np.mean(kur)/i)
    kurtosis_values = np.array(kurtosis_values)
    plt.figure()
    plt.plot(np.arange(1,30),kurtosis_values)
    plt.xlabel('Components')
    plt.ylabel('Normalized Mean Kurtosis')
    plt.grid()
    plt.title('Normalized Mean Kurtosis vs Components for Bldg')
    plt.savefig('bldg_ica_kurtosis.png')
    
    #plot ICA 2D
    ica_2D = FastICA(n_components=2)
    x_ICA_2D = ica_2D.fit_transform(X_train_ICA)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('ICA 1st Component')
    ax.set_ylabel('ICA 2nd Component')
    for i in range(len(Y_train)):
        if Y_train[i] == 0:
            ax.scatter(x_ICA_2D[i, :][0], x_ICA_2D[i, :][1], c = 'b', marker='o', label='0')
        elif Y_train[i] == 1:
            ax.scatter(x_ICA_2D[i, :][0], x_ICA_2D[i, :][1], c = 'y', marker='o', label='1')
    ax.grid()
    plt.title('Bldg Data Reduced to 2D by ICA')
    plt.savefig('bldg_ica_2d.png')
    plt.show()
    
    #plot ICA 3D
    ica_3D = FastICA(n_components = 3)
    x_ICA_3D = ica_3D.fit_transform(X_train_ICA)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(Y_train)):
        if Y_train[i] == 0:
            ax.scatter(x_ICA_3D[i, :][0], x_ICA_3D[i, :][1], x_ICA_3D[i, :][2], c = 'b', marker='o', label='0')
        elif Y_train[i] == 1:
            ax.scatter(x_ICA_3D[i, :][0], x_ICA_3D[i, :][1], x_ICA_3D[i, :][2], c = 'y', marker='o', label='1')
    ax.set_xlabel('ICA 1st Component')
    ax.set_ylabel('ICA 2nd Component')
    ax.set_zlabel('ICA 3rd Component')
    plt.title('Bldg Data Reduced to 3D by ICA')
    plt.savefig('bldg_ica_3d.png')
    plt.show()
    #best ICA X
    X_ICA_best = FastICA(n_components = 2).fit_transform(X_train_ICA)
    
    #RP
    #plot RP errors
    X_train_RP = StandardScaler().fit_transform(X_train)
    num_iter = 100
    num_features = 14
    reconstruction_error = []
    reconstruction_variance = []
    for i in range(1,num_features):
        mmse = []
        for j in range(0,num_iter):
            grp = GaussianRandomProjection(n_components=i)
            X_RP = grp.fit(X_train_RP)
            w = X_RP.components_
            p = pinv(w)
            reconstructed = (p.dot(w).dot(X_train_RP.T)).T
            mmse.append(mean_squared_error(X_train_RP,reconstructed))
        reconstruction_variance.append(np.std(mmse))
        reconstruction_error.append(np.mean(mmse))    
    reconstruction_error = np.array(reconstruction_error)
    plt.plot(np.arange(1,num_features),reconstruction_error)
    plt.fill_between(np.arange(1,num_features),reconstruction_error-reconstruction_variance, reconstruction_error+reconstruction_variance, alpha=0.2)
    plt.xlabel('Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction error vs Number of Components for Bldg')
    plt.grid()
    plt.savefig('bldg_rp_error.png')
    plt.show()
    
    #plot RP 2D
    X_RP_2D = GaussianRandomProjection(n_components = 2).fit_transform(X_train_RP)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('RP 1st Component')
    ax.set_ylabel('RP 2nd Component')
    for i in range(len(Y_train)):
        if Y_train[i] == 0:
            ax.scatter(X_RP_2D[i, :][0], X_RP_2D[i, :][1], c = 'b', marker='o', label='0')
        elif Y_train[i] == 1:
            ax.scatter(X_RP_2D[i, :][0], X_RP_2D[i, :][1], c = 'r', marker='o', label='1')
    ax.grid()
    plt.title('Bldg Data Reduced to 2D by RP')
    plt.savefig('bldg_rp_2d.png')
    plt.show()
    
    #plot RP 3D
    X_RP_3D = GaussianRandomProjection(n_components = 3).fit_transform(X_train_RP)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(Y_train)):
        if Y_train[i] == 0:
            ax.scatter(X_RP_3D[i, :][0], X_RP_3D[i, :][1], X_RP_3D[i, :][2], c = 'b', marker='o', label='0')
        elif Y_train[i] == 1:
            ax.scatter(X_RP_3D[i, :][0], X_RP_3D[i, :][1], X_RP_3D[i, :][2], c = 'r', marker='o', label='1')
    ax.set_xlabel('RP 1st Component')
    ax.set_ylabel('RP 2nd Component')
    ax.set_zlabel('RP 3rd Component')
    plt.title('Bldg Data Reduced to 3D by RP')
    plt.savefig('bldg_rp_3d.png')
    plt.show()
    #best RP X
    X_RP_best = GaussianRandomProjection(n_components = 3).fit_transform(X_train_RP)
    
    #FA
    X_train_FA = StandardScaler().fit_transform(X_train)
    X_FA_2D = FeatureAgglomeration(n_clusters=2).fit_transform(X_train_FA)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('FA 1st Component')
    ax.set_ylabel('FA 2nd Component')
    for i in range(len(Y_train)):
        if Y_train[i] == 0:
            ax.scatter(X_FA_2D[i, :][0], X_FA_2D[i, :][1], c = 'b', marker='o', label='0')
        elif Y_train[i] == 1:
            ax.scatter(X_FA_2D[i, :][0], X_FA_2D[i, :][1], c = 'r', marker='o', label='1')
    ax.grid()
    plt.title('Bldg Data Reduced to 2D by FA')
    plt.savefig('bldg_fa_2d.png')
    plt.show()
    #best FA X
    X_FA_best = FeatureAgglomeration(n_clusters=2).fit_transform(X_train_FA)
    
    #Kmeans on PCA data
    range_n_clusters = np.arange(2,30,1)
    silhouette = []
    for n_clusters in range_n_clusters:
        k_means_clustering = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        k_means_clustering.fit(X_PCA_best)
        silhouette.append(silhouette_score(X_PCA_best, k_means_clustering.labels_, metric="euclidean"))
    

    silhouette = np.array(silhouette)    
    plt.figure()
    plt.plot(range_n_clusters,silhouette)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette')
    plt.title('Clusters vs Silhouette for bldg')
    plt.grid()
    plt.savefig('bldg_kmeans_pca_silhouette.png')
    
    #Kmeans on ICA data
    silhouette = []
    for n_clusters in range_n_clusters:
        k_means_clustering = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        k_means_clustering.fit(X_ICA_best)
        silhouette.append(silhouette_score(X_ICA_best, k_means_clustering.labels_, metric="euclidean"))
    
    silhouette = np.array(silhouette)    
    plt.figure()
    plt.plot(range_n_clusters,silhouette)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette')
    plt.title('Clusters vs Silhouette for bldg')
    plt.grid()
    plt.savefig('bldg_kmeans_ica_silhouette.png')
    
    #Kmeans on RP data
    silhouette = []
    for n_clusters in range_n_clusters:
        k_means_clustering = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        k_means_clustering.fit(X_RP_best)
        silhouette.append(silhouette_score(X_RP_best, k_means_clustering.labels_, metric="euclidean"))

    silhouette = np.array(silhouette)    
    plt.figure()
    plt.plot(range_n_clusters,silhouette)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette')
    plt.title('Clusters vs Silhouette for bldg')
    plt.grid()
    plt.savefig('bldg_kmeans_rp_silhouette.png')
    
    #Kmeans on FA data
    silhouette = []
    for n_clusters in range_n_clusters:
        k_means_clustering = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        k_means_clustering.fit(X_FA_best)
        silhouette.append(silhouette_score(X_FA_best, k_means_clustering.labels_, metric="euclidean"))

    silhouette = np.array(silhouette)    
    plt.figure()
    plt.plot(range_n_clusters,silhouette)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette')
    plt.title('Clusters vs Silhouette for bldg')
    plt.grid()
    plt.savefig('bldg_kmeans_fa_silhouette.png')
        
    k = 4
    k_means_clustering = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    k_means_clustering.fit(X_train)    
    plt.figure()
    plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, k))
    plt.xlabel('Cluster')
    plt.ylabel('Samples per Cluster')
    plt.title('Distribution of data per cluster for K-Means')
    plt.grid()
    plt.savefig('bldg_kmeans_distribution.png')
    
    k_means_clustering = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    k_means_clustering.fit(X_PCA_best)    
    plt.figure()
    plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, k))
    plt.xlabel('Cluster')
    plt.ylabel('Samples per Cluster')
    plt.title('Distribution of data per cluster for K-Means (PCA)')
    plt.grid()
    plt.savefig('bldg_kmeans_distribution_pca.png')
    
    k_means_clustering = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    k_means_clustering.fit(X_ICA_best)    
    plt.figure()
    plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, k))
    plt.xlabel('Cluster')
    plt.ylabel('Samples per Cluster')
    plt.title('Distribution of data per cluster for K-Means (ICA)')
    plt.grid()
    plt.savefig('bldg_kmeans_distribution_ica.png')
    
    k_means_clustering = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    k_means_clustering.fit(X_RP_best)    
    plt.figure()
    plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, k))
    plt.xlabel('Cluster')
    plt.ylabel('Samples per Cluster')
    plt.title('Distribution of data per cluster for K-Means (RP)')
    plt.grid()
    plt.savefig('bldg_kmeans_distribution_rp.png')
    
    k_means_clustering = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    k_means_clustering.fit(X_FA_best)    
    plt.figure()
    plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, k))
    plt.xlabel('Cluster')
    plt.ylabel('Samples per Cluster')
    plt.title('Distribution of data per cluster for K-Means (FA)')
    plt.grid()
    plt.savefig('bldg_kmeans_distribution_fa.png')
    
    #EM on PCA
    n_components_range = range(1, 30)
    X_train_1 = StandardScaler().fit_transform(X_train)
    trainData = [X_train_1, X_PCA_best, X_ICA_best, X_RP_best, X_FA_best]
    bic = np.zeros((len(trainData),len(n_components_range)))
    for i, X_train in enumerate(trainData):
        for j, n_components in enumerate(n_components_range):
            gmm = GaussianMixture(n_components=n_components, covariance_type='full')
            gmm.fit(X_train)
            bic[i][j] = gmm.bic(X_train)
    #plot BIC for EM
    plt.figure()
    plt.plot(n_components_range, bic[0], label = 'original')
    plt.plot(n_components_range, bic[1], label = 'PCA')
    plt.plot(n_components_range, bic[2], label = 'ICA')
    plt.plot(n_components_range, bic[3], label = 'RP')
    plt.plot(n_components_range, bic[4], label = 'FA')
    plt.legend()
    plt.xticks(np.arange(0, len(n_components_range)+1, 4.0))
    plt.title("Number of Components vs BIC for Bldg")
    plt.xlabel("Number of Components")
    plt.ylabel("BIC")
    plt.savefig('bldg_gmm_bic_differnt_data.png')
    plt.show()
    
    # #NN on the reduced X
    param_kfold = 10 #parameter k for kfold CV
    param_num_neuron = 8 #set number of neurons on one layer
    param_layers = (param_num_neuron,param_num_neuron,param_num_neuron,param_num_neuron)#set layer structure
    
    #setting figsize and axes
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))        
    #NN
    X_train = X_PCA_best
    title = "Learning Curves NN, relu"
    cv = cv = KFold(n_splits=param_kfold)   
    estimator = MLPClassifier(hidden_layer_sizes=param_layers, activation='relu',random_state=RANDOM_STATE)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    
    title = "Learning Curves NN, identity"
    cv = cv = KFold(n_splits=param_kfold)   
    estimator = MLPClassifier(hidden_layer_sizes=param_layers, activation='identity',random_state=RANDOM_STATE)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    
    title = "Learning Curves NN, tanh"
    cv = cv = KFold(n_splits=param_kfold)   
    estimator = MLPClassifier(hidden_layer_sizes=param_layers, activation='tanh',random_state=RANDOM_STATE, max_iter=300)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    
    title = "Learning Curves NN, logistric"
    cv = cv = KFold(n_splits=param_kfold)   
    estimator = MLPClassifier(hidden_layer_sizes=param_layers, activation='logistic',random_state=RANDOM_STATE, max_iter=300)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,3], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    
    plt.savefig('bldg_pca_nn.png')
    
    #NN with the clustered data
    param_kfold = 10 #parameter k for kfold CV
    param_num_neuron = 8 #set number of neurons on one layer
    param_layers = (param_num_neuron,param_num_neuron,param_num_neuron,param_num_neuron)#set layer structure
    
    #setting figsize and axes
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))        
    X_train = KMeans(n_clusters=13, random_state=RANDOM_STATE).fit_transform(X_PCA_best)
    title = "Learning Curves NN, relu"
    cv = cv = KFold(n_splits=param_kfold)   
    estimator = MLPClassifier(hidden_layer_sizes=param_layers, activation='relu',random_state=RANDOM_STATE)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    
    title = "Learning Curves NN, identity"
    cv = cv = KFold(n_splits=param_kfold)   
    estimator = MLPClassifier(hidden_layer_sizes=param_layers, activation='identity',random_state=RANDOM_STATE)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    
    title = "Learning Curves NN, tanh"
    cv = cv = KFold(n_splits=param_kfold)   
    estimator = MLPClassifier(hidden_layer_sizes=param_layers, activation='tanh',random_state=RANDOM_STATE, max_iter=300)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    
    title = "Learning Curves NN, logistric"
    cv = cv = KFold(n_splits=param_kfold)   
    estimator = MLPClassifier(hidden_layer_sizes=param_layers, activation='logistic',random_state=RANDOM_STATE, max_iter=300)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,3], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    
    plt.savefig('bldg_pca_km_nn.png')
    

if __name__ == "__main__":
	main()
	
