###Notebook to select portfolio optimized mix of nodes
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab

def select_nodes(full_node_dataset, current_node_dataset, num_target_nodes, number_of_clusters, run_range_values, cluster_range_to_run, target_node_range_to_run, iterations, show_diagram,iso):
    iso = iso.upper()
    full_node_dataset_corr = full_node_dataset.corr()
    current_node_dataset_corr = current_node_dataset.corr()

    Z = linkage(full_node_dataset_corr, "average")

    if show_diagram:
        # Plot linkage diagram
        plt.figure(figsize=(20, 10))
        labelsize = 10
        ticksize = 6
        plt.title(iso + ' Hierarchical Clustering Dendrogram', fontsize=20)
        plt.xlabel('Node', fontsize=20)
        plt.ylabel('Distance', fontsize=20)
        dendrogram(Z,
                   leaf_rotation=90.,  # rotates the x axis labels
                   leaf_font_size=1,  # font size for the x axis labels
                   labels=full_node_dataset_corr.columns)
        pylab.yticks(fontsize=ticksize)
        pylab.xticks(rotation=-90, fontsize=ticksize)
        plt.show()

    if run_range_values == True:
        corr_to_num_nodes_df = pd.DataFrame(columns=['NumClusters', 'NumNodes', "AvgCorrelation"])

        for num_target_nodes in target_node_range_to_run:

            for number_of_clusters in cluster_range_to_run:
                print('Running '+ str(number_of_clusters) + ' Clusters and ' + str(num_target_nodes) + ' Nodes')
                num_node_iter = 0
                node_clusters = pd.DataFrame(zip(full_node_dataset.columns[1:], fcluster(Z, number_of_clusters, criterion='maxclust')),columns=['Node', 'ClusterNumber'])
                sample_node_df = pd.DataFrame()
                new_node_avg_corr = 1.0
                target_nodes_per_cluster = int(round(num_target_nodes / number_of_clusters, 0))

                for iter in range(iterations):
                    clustered_node_df = pd.DataFrame()

                    for cluster in range(1, number_of_clusters + 1, 1):

                        if cluster not in node_clusters['ClusterNumber'].values: break  ## iterate through each cluster and take an equal amount of nodes from each. If there are not enough nodes in that cluster, add the deficient number of nodes to the remaining clusters
                        cluster_subset = node_clusters[node_clusters['ClusterNumber'] == cluster]

                        if target_nodes_per_cluster <= len(cluster_subset):
                            random_sample_nodes = cluster_subset.sample(n=target_nodes_per_cluster, replace=False)
                            clustered_node_df = clustered_node_df.append(random_sample_nodes)
                        else:
                            clustered_node_df = clustered_node_df.append(cluster_subset)

                        if len(clustered_node_df) < num_target_nodes:
                            remaining_nodes = num_target_nodes - len(clustered_node_df)
                            remaining_clusters = number_of_clusters - cluster
                            if remaining_clusters > 0:
                                target_nodes_per_cluster = int(round(remaining_nodes / remaining_clusters, 0))
                            else:
                                target_nodes_per_cluster = 0

                    sample_nodes_df = full_node_dataset[clustered_node_df['Node']]
                    sample_node_corr = sample_nodes_df.corr()

                    sample_node_avg_corr = round(
                        sample_node_corr.values[np.triu_indices_from(sample_node_corr.values, 1)].mean(), 4)

                    if sample_node_avg_corr < new_node_avg_corr:
                        new_node_avg_corr = sample_node_avg_corr
                        new_node_df = sample_node_df
                        new_node_corr = sample_node_corr
                    num_node_iter += 1

                corr_to_num_nodes_df = corr_to_num_nodes_df.append({'NumClusters': number_of_clusters, 'NumNodes': num_target_nodes, 'AvgCorrelation': new_node_avg_corr}, ignore_index=True)


        corr_to_num_nodes_df.to_excel(iso + '_mean_corr_by_num_nodes.xlsx')

        for cluster_group in corr_to_num_nodes_df['NumClusters'].unique():
            graph_df = corr_to_num_nodes_df[corr_to_num_nodes_df['NumClusters']==cluster_group]
            plt.plot(graph_df['NumNodes'].values, graph_df['AvgCorrelation'].values)

        legend_values = [' Cluster ' + str(cluster) for cluster in corr_to_num_nodes_df['NumClusters'].unique()]
        plt.legend(legend_values)

        plt.title(iso + ' Correlation by Number of Clusters and Nodes. '+str(iterations)+ ' Iterations')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Average Correlation')
        plt.xticks(graph_df['NumNodes'].values)
        plt.show()
        exit()




    else:
        num_node_iter = 0
        node_clusters = pd.DataFrame(zip(full_node_dataset.columns[1:], fcluster(Z, number_of_clusters, criterion='maxclust')),columns=['Node', 'ClusterNumber'])
        sample_node_df = pd.DataFrame()
        new_node_df = pd.DataFrame()
        new_node_avg_corr = 1.0
        target_nodes_per_cluster = int(round(num_target_nodes / number_of_clusters, 0))
        cluster_count_df = node_clusters.groupby(['ClusterNumber'])['Node'].count().reset_index()
        cluster_count_df.columns = ['Cluster Number', 'Count of Nodes in Cluster']
        print(cluster_count_df)
        print()

        for iter in range(iterations):
            clustered_node_df = pd.DataFrame()

            for cluster in range(1, number_of_clusters + 1, 1):

                if cluster not in node_clusters['ClusterNumber'].values: break  ## iterate through each cluster and take an equal amount of nodes from each. If there are not enough nodes in that cluster, add the deficient number of nodes to the remaining clusters
                cluster_subset = node_clusters[node_clusters['ClusterNumber'] == cluster]

                if target_nodes_per_cluster <= len(cluster_subset):
                    random_sample_nodes = cluster_subset.sample(n=target_nodes_per_cluster, replace=False)
                    clustered_node_df = clustered_node_df.append(random_sample_nodes)
                else:
                    clustered_node_df = clustered_node_df.append(cluster_subset)

                if len(clustered_node_df) < num_target_nodes:
                    remaining_nodes = num_target_nodes - len(clustered_node_df)
                    remaining_clusters = number_of_clusters - cluster
                    if remaining_clusters >0:
                        target_nodes_per_cluster = int(round(remaining_nodes / remaining_clusters, 0))
                    else: target_nodes_per_cluster = 0

            sample_nodes_df = full_node_dataset[clustered_node_df['Node']]
            sample_node_corr = sample_nodes_df.corr()

            sample_node_avg_corr = round(sample_node_corr.values[np.triu_indices_from(sample_node_corr.values, 1)].mean(), 4)

            if sample_node_avg_corr < new_node_avg_corr:
                new_node_avg_corr = sample_node_avg_corr
                new_node_df = clustered_node_df
                new_node_corr = sample_node_corr

            num_node_iter += 1

        full_node_avg_corr = round(full_node_dataset_corr.values[np.triu_indices_from(full_node_dataset_corr.values, 1)].mean(),4)
        current_node_avg_corr = round(current_node_dataset_corr.values[np.triu_indices_from(current_node_dataset_corr.values, 1)].mean(),4)

        plt.figure(figsize=(14, 7))
        ax1 = plt.subplot(1, 3, 1)
        ax1.matshow(current_node_dataset_corr, cmap=cm.get_cmap('coolwarm'), vmin=0, vmax=1)
        plt.title('Current ' + iso + ' Traded Nodes Corr')
        plt.text(0, len(current_node_dataset_corr) * 1.05, 'Avg Corr= ' + str(current_node_avg_corr))

        ax2 = plt.subplot(1, 3, 2)
        ax2.matshow(full_node_dataset_corr, cmap=cm.get_cmap('coolwarm'), vmin=0, vmax=1)
        plt.title('All ' + iso + ' Tradeable Nodes Corr')
        plt.text(0, len(full_node_dataset_corr) * 1.05, 'Avg Corr= ' + str(full_node_avg_corr))

        ax3 = plt.subplot(1, 3, 3)
        ax3.matshow(new_node_corr, cmap=cm.get_cmap('coolwarm'), vmin=0, vmax=1)
        plt.title('New ' + iso + ' Traded Nodes Corr')
        plt.text(0, len(new_node_corr) * 1.05, 'Avg Corr= ' + str(new_node_avg_corr))

        plt.show()

        xl_writer = pd.ExcelWriter(iso + '_corrMatrixes.xlsx', engine='xlsxwriter')
        pd.DataFrame(current_node_dataset_corr).to_excel(xl_writer, sheet_name='Original')
        pd.DataFrame(full_node_dataset_corr).to_excel(xl_writer, sheet_name='Full')
        pd.DataFrame(new_node_corr).to_excel(xl_writer, sheet_name='New')
        xl_writer.save()

        new_node_df.to_csv(iso+'_selected_nodes.csv')
        node_clusters.to_csv(iso+'_all_node_clusters.csv')

    return new_node_df, new_node_corr

###################################################################################################

#for iso in ['pjm','miso','spp','isone']:
    #iso = iso
    #total nodes: isone:621  spp:502   pjm:576   miso:1840

iso = 'isone'
num_target_nodes = 30 # Num Nodes if Known  pjm:110  miso:150 spp:90 isone:30   ercot:20
number_of_clusters = 6 # Num Clusters if Known  pjm:20   miso:12  spp:8   isone:6   ercot:6
iterations = 100  # Num Iterations to Reduce Random Sampling
show_diagram = True  # True if you want to see the tree diagram

run_range_values = False #True if creating a list of lowest correlations according to the number of target nodes and clusters
cluster_range_to_run = range(6,7,3)
target_node_range_to_run = range(10,160,10)

full_node_dataset = pd.read_csv(iso+'_darts_wide_DSTadjusted.csv', index_col=['Date'])
current_nodes = pd.read_excel(iso+'_current_nodes.xlsx')
current_nodes_dataset = full_node_dataset[current_nodes['NODES']]
selected_nodes_df, new_node_corr = select_nodes(full_node_dataset=full_node_dataset,
                                                current_node_dataset=current_nodes_dataset,
                                                num_target_nodes=num_target_nodes,
                                                number_of_clusters = number_of_clusters,
                                                run_range_values=run_range_values,
                                                cluster_range_to_run = cluster_range_to_run,
                                                target_node_range_to_run = target_node_range_to_run,
                                                iterations=iterations,
                                                show_diagram = show_diagram,
                                                iso=iso)






