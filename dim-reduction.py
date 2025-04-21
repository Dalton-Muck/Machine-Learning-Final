from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas
import os

data = pandas.read_csv('./results/selected_features_0.96.csv')

#Contains all the data from the dataset aside from the names of the labels (Also known as 'Targets')
X = data.drop(columns = ['target'])

#Have the y-axis data be the labels.
y = data['target']

#Make a umap figure
umapFig = UMAP(n_components=2, random_state=42)
#Fit the umap with the data from above
reducedData1 = umapFig.fit_transform(X)

#Make a PCA figure
pcaFig = PCA(n_components = 2)
#Fit the PCA with the data from above
reducedData2 = pcaFig.fit_transform(X)

#Make a TSNE figure
tsneFig = TSNE(n_components = 2)
#Fit the TSNE with the data from above
reducedData3 = tsneFig.fit_transform(X)

os.makedirs("figures/dim-reduce", exist_ok=True)

plt.figure()
#X axis is the first dimension of the fitted umap data and the y axis is the second dimension. c is used for coloring the data/labels
umapPlot = plt.scatter(reducedData1[:, 0], reducedData1[:,1], c=y)

#Adding things like labels, a title, a grid, and a legend
plt.title("UMAP Visualization")
plt.xlabel("First Dimension")
plt.ylabel("Second Dimension")
plt.grid(True)
plt.savefig('./figures/dim-reduce/umap_visualization.png')
plt.close()


plt.figure()
#X axis is the first dimension of the fitted umap data and the y axis is the second dimension. c is used for coloring the data/labels
pcaPlot = plt.scatter(reducedData2[:, 0], reducedData2[:,1], c=y)

#Adding things like labels, a title, a grid, and a legend
plt.title("PCA Visualization")
plt.xlabel("First Dimension")
plt.ylabel("Second Dimension")
plt.grid(True)
plt.savefig('./figures/dim-reduce/pca_visualization.png')
plt.close()


plt.figure()
#X axis is the first dimension of the fitted umap data and the y axis is the second dimension. c is used for coloring the data/labels
tsnePlot = plt.scatter(reducedData3[:, 0], reducedData3[:,1], c=y)

#Adding things like labels, a title, a grid, and a legend
plt.title("TSNE Visualization")
plt.xlabel("First Dimension")
plt.ylabel("Second Dimension")
plt.grid(True)
plt.savefig('./figures/dim-reduce/tsne_visualization.png')
plt.close()

#Covert model data to csv files
umapDF = pandas.DataFrame(reducedData1, columns = ['Dimension 1', 'Dimension 2'])
umapDF['Target'] = y.values
umapDF.to_csv('./results/umap_reduced_data.csv', index=False)

pcaDF = pandas.DataFrame(reducedData2, columns = ['Dimension 1', 'Dimension 2'])
pcaDF['Target'] = y.values
pcaDF.to_csv('./results/pca_reduced_data.csv', index=False)

tsneDF = pandas.DataFrame(reducedData3, columns = ['Dimension 1', 'Dimension 2'])
tsneDF['Target'] = y.values
tsneDF.to_csv('./results/tsne_reduced_data.csv', index=False)
