import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
import glob

from keras.preprocessing import image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input

def run_pretrained_model(model, target_size=224):
    """
    
    Parameters
    ----------
    model : keras.applications.model
        The keras.applications.model object
    target_size : int, optional
        optional shape tuple
    
    Returns
    -------
    numpy.array
        pre-trained feature list
    """
    pretrain_list = []
    for filename in glob.glob('images/*.jpg'):
        img = image.load_img(filename, target_size=(target_size, target_size))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        pretrain_nparray = np.array(model.predict(img_data))
        pretrain_list.append(pretrain_nparray.flatten())
    return np.array(pretrain_list)

def scalez_pca(df, scalez=True, PCA=True, n_components=None, batch_size=None):
    """apply standardization and IncrementalPCA to the input dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    scalez : bool, optional
        Whether to process standardization
    PCA : bool, optional
        Whether to process PCA
    n_components : int or float, optional
        If 0 < n_components < 1, select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
    batch_size : int or None, optional
        The number of samples to use for each batch. Only used when calling fit. If batch_size is None, then batch_size is inferred from the data and set to 5 * n_features, to provide a balance between approximation accuracy and memory consumption.
    
    Returns
    -------
    pd.DataFrame
        the transformed dataframe
    """
    if scalez:
        sc = StandardScaler()
        sc.fit(df)
        df = pd.DataFrame(data=sc.transform(df), index=df.index, columns=df.columns)
    if PCA:
        pca = IncrementalPCA(n_components=min(len(df), round(len(df.columns)*n_components) if n_components < 1 else n_components), batch_size=batch_size)
        pca.fit(df)
        df = pd.DataFrame(data=pca.transform(df), index=df.index, columns=['pca_'+str(i+1) for i in range(0, pca.n_components_)])
        print(pca.explained_variance_ratio_.sum())
        print(df.shape)
    return df

def minibatchkmeans(df, cluster_number_list=None, final_cluster_number=None, batch_size=1000, max_iter=1000, max_no_improvement=100, n_init=3, random_state=None, show_plot=True):
    """
    minibatchkmeans function
    
    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    cluster_number_list : list or None, optional
        list of clustering number to try
    final_cluster_number : int or None, optional
        final cluster number for output dataframe
    batch_size : int, optional, default: 100
        Size of the mini batches.
    max_iter : int, optional
        Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.
    max_no_improvement : int, default: 10
        Control early stopping based on the consecutive number of mini batches that does not yield an improvement on the smoothed inertia.
    n_init : int, optional
        [description] (the default is 3, which [default_description])
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and random reassignment. Use an int to make the randomness deterministic. See Glossary.
    show_plot : bool, optional
        Whether to show the plotting
    
    Returns
    -------
    pd.DataFrame
        The dataframe with added 'cluster' column for the running the minibatch kmean with final_cluster_number
    """
    cluster_errors = []
    silhouette_scores = []
    if cluster_number_list is not None:
        for cluster_number in cluster_number_list:
            cluster = MiniBatchKMeans(cluster_number, random_state=random_state, batch_size=batch_size, max_iter=max_iter, max_no_improvement=max_no_improvement, n_init=n_init)
            cluster.fit(df)
            cluster_errors.append(cluster.inertia_)
            silhouette_scores.append(silhouette_score(df, cluster.labels_, metric='euclidean'))
            print('inertia of #'+str(cluster_number)+':', cluster.inertia_)
            print('silhouette_score of #'+str(cluster_number)+':', silhouette_score(df, cluster.labels_, metric='euclidean'))

        if show_plot:
            clusters_df = pd.DataFrame({ "num_clusters": cluster_number_list, "cluster_errors": cluster_errors, 'silhouette_score': silhouette_scores})
            plt.figure(figsize=(12,6))
            plt.title('Inertia')
            plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )

            plt.figure(figsize=(12,6))
            plt.title('Silhouette Coefficient')
            plt.plot(clusters_df.num_clusters, clusters_df.silhouette_score, marker = "o" )
    
    if final_cluster_number is not None:
        final_cluster = MiniBatchKMeans(final_cluster_number, random_state=random_state, batch_size=batch_size, max_iter=max_iter, max_no_improvement=max_no_improvement, n_init=n_init)
        final_cluster.fit(df)
        print('final inertia:', final_cluster.inertia_)
        print('final silhouette_score:', silhouette_score(df, final_cluster.labels_, metric='euclidean'))
        return pd.DataFrame({'cluster': final_cluster.predict(df) + 1}, index=df.index)

def main(model, target_size, scalez=True, PCA=True, n_components=None, pca_batch_size=None, cluster_number_list=None, final_cluster_number=None, minikmean_batch_size=1000, max_iter=1000, max_no_improvement=100, n_init=3, random_state=None, show_plot=True):
    """
    The main function to combine the above functions
    """
    np_list = run_pretrained_model(model, target_size=target_size)
    df = pd.DataFrame(np_list, index=range(0, len(np_list)))
    print(df.shape)
    df_after = scalez_pca(df, scalez=scalez, PCA=PCA, n_components=n_components, batch_size=pca_batch_size)
    minibatchkmeans(df_after, cluster_number_list=cluster_number_list, final_cluster_number=final_cluster_number, batch_size=minikmean_batch_size, max_iter=max_iter, max_no_improvement=max_no_improvement, n_init=n_init, random_state=random_state, show_plot=show_plot)
    return df_after

random_state = 2
df_list = []
for pooling in [None, 'max', 'avg']:
    for target_size in [32, 48, 64, 80]:
        for k, v in {
            'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(target_size, target_size, 3), pooling=pooling)
            ,'VGG19':  VGG19(weights='imagenet', include_top=False, input_shape=(target_size, target_size, 3), pooling=pooling)
            ,'ResNet50':  ResNet50(weights='imagenet', include_top=False, input_shape=(target_size, target_size, 3), pooling=pooling)
        }.items():
            for pca in [.5, .75]:
                print(i, pooling, '\n'+str(target_size), '\n'+str(k), '\n'+str(pca))
                df = main(model=v, target_size=target_size, scalez=True, PCA=True, n_components=pca, pca_batch_size=None, cluster_number_list=[20, 40], final_cluster_number=None, minikmean_batch_size=1000, max_iter=1000, max_no_improvement=100, n_init=5, random_state=random_state, show_plot=False)
                df_list.append(df)

#choose target_size=80, pooling='max', pca n_components=0.5 explained_variance, which have lowest inertia
final_model = VGG16(weights='imagenet', include_top=False, input_shape=(80, 80, 3), pooling='max')
transformed_df = main(model=final_model,target_size=80, n_components=.5, cluster_number_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], random_state=random_state, minikmean_batch_size=1000, max_iter=1000, max_no_improvement=100, n_init=10)

#choose final cluster number as 80 with lower inertia score but with larger silhouette_score
result_df = minibatchkmeans(transformed_df, final_cluster_number=80, batch_size=1000, max_iter=50000, max_no_improvement=5000, n_init=3, random_state=random_state)

result_df_list = []
for i in range(1, 81):
    result_df_list.append(pd.DataFrame({'Cluster '+str(i): ["'"+str(index).zfill(5)+"'" for index in result_df[result_df['cluster'] == i].index]}))
submissin_df = pd.concat(result_df_list, ignore_index=True, axis=1)
submissin_df = submissin_df.rename(columns={col: 'Cluster '+str(col+1) for col in submissin_df.columns}).fillna('')
submissin_df.to_csv('A3_kkhuaa_20123133_prediction.csv', index=False)
