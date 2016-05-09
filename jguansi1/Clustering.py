# Name: Jerson Guansing
# Project 4
# CMSC 471
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

# read the data file
def readFile(fileName):
	X = []
	with open(fileName, "r") as f:
		for line in f:
			line = [ float(i) for i in line.split() ]
			if len(line) == 2:
				X.append(line)
	X = np.array(X)
	return X

# cluster the data point to a 
def cluster_points(X, mu):
	clusters = {}
	for x in X:
		bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
		try:
			# append x to the mu key cluster
			clusters[bestmukey].append(x)
		except KeyError:
			# mu key not in cluster yet, so create it
			clusters[bestmukey] = [x]
	return clusters

# readjust the center of clusters to the mean
def reevaluate_centers(mu, clusters):
	newmu = []
	keys = sorted(clusters.keys())
	for k in keys:
		newmu.append(np.mean(clusters[k], axis = 0))
	return newmu

# check if the current center is arithmetically the mean of the cluster
def has_converged(mu, oldmu):
	return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

# cluster the data by finding the centers
def find_centers(X, K):
	# Initialize to K random centers
	oldmu = random.sample(list(X), K)
	mu = random.sample(list(X), K)
	while not has_converged(mu, oldmu):
		oldmu = mu
		# Assign all points in X to clusters
		clusters = cluster_points(X, mu)
		# Reevaluate centers
		mu = reevaluate_centers(oldmu, clusters)
	return (mu, clusters)

# plot the k-means clustered data
def plotData(data, k, n):
	## iterate through the k clusters
	for i in data[1]:
		cluster = data[1][i]
		X, Y = [ data[0][i][0] ], [ data[0][i][1] ]
		current_color = np.random.rand(3,)
		for points in cluster:
			X.append(points[0])
			Y.append(points[1])
			# draw a line between the cluster's center and its clustered data points
			plt.plot([data[0][i][0], points[0]], [data[0][i][1], points[1] ], color = current_color)
		# create a scatter plot of all the data points in the cluster
		plt.scatter(X, Y, color = current_color )
	plt.title("K-Means Clustering\n k = " + str(k) + " clusters   n = " + str(n) + " data points")
	plt.show()

def main(argv):
	if len(argv) != 3:
		print("The program expects two (2) arguments.")
		print("Clustering.py <number of clusters> <filename>")
	else:
		k = int(argv[1])
		print("Reading the data file...")
		X = readFile(argv[2])
		print("Performing k-means clustering...")
		data = find_centers(X, k)
		print("Generating graph...")
		plotData(data, k, len(X))
	
main(sys.argv)
