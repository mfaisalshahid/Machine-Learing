import math
import random
import time
from scipy.spatial.distance import cosine, jaccard, euclidean
from scipy.stats import mode
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import normalize

######################################################################
# This section contains functions for loading CSV (comma separated values)
# files and convert them to a dataset of instances.
# Each instance is a tuple of attributes. The entire dataset is a list
# of tuples.
######################################################################

# Loads a CSV files into a list of tuples.
# Ignores the first row of the file (header).
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
#   fileName: name of the CSV file to be read
# Returns: a list of tuples
def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close() # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    return dataset

# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    cleanLine = line.strip()
    # get rid of quotes
    cleanLine = cleanLine.replace('"', '')
    # separate the fields
    lineList = cleanLine.split(",")
    # convert strings into numbers
    stringsToNumbers(lineList)
    # Move last item to first
    lineList.insert(0, lineList.pop())
    lineTuple = tuple(lineList)
    return lineTuple

# Destructively converts all the string elements representing numbers
# to floating point numbers.
# Parameters:
#   myList: a list of strings
# Returns None
def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])

# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
  if len(s) == 0:
    return False
  if  len(s) > 1 and s[0] == "-":
      s = s[1:]
  for c in s:
    if c not in "0123456789.":
      return False
  return True


######################################################################
# This section contains functions for clustering a dataset
# using the k-means algorithm.
######################################################################

def distance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    u = list(instance1[1:])
    v = list(instance2[1:])
    if dist_metric == "Cosine":
        return cosine(u, v)
    elif dist_metric == "Jaccard":
        return jaccard(u, v)
    else: #dist_metric == "Euclidean": #Default
        return euclidean(u, v)


def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)

def assign(instance, centroids):
    minDistance = distance(instance, centroids[0])
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i])
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex

def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assignAll(instances, centroids):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids)
        clusters[clusterIndex].append(instance)
    return clusters

def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids

def kmeans(instances, k, initCentroids=None):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids
    prevCentroids = []
    iteration = 0
    while (centroids != prevCentroids):
        iteration += 1
        clusters = assignAll(instances, centroids)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids)
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["iterations"] = iteration
    return result

def kmeans_maxIter(instances, k):
    result = {}
    centroids = random.sample(instances, k)
    iteration = 0
    while (iteration < 100):
        iteration += 1
        clusters = assignAll(instances, centroids)
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids)
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["iterations"] = iteration
    return result

def kmeans_minSS(instances, k):
    result = {}
    currSS = 0
    notFoundMin = True
    iteration = 0
    centroids = random.sample(instances, k)
    while (notFoundMin):
        iteration += 1
        clusters = assignAll(instances, centroids)
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids)
        if withinss < currSS or currSS == 0:
            currSS = withinss
        else:
            notFoundMin = False

    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["iterations"] = iteration
    return result

def computeWithinss(clusters, centroids):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += distance(centroid, instance) ** 2
    return result

# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss
def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")
    for i in range(1, n+1):
        # print("k-means trial %d," % i)
        trialClustering = kmeans(instances, k)
        print("withinss: %.1f" % trialClustering["withinss"])
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    print("Trial with minimum withinss:", minWithinssTrial)
    return bestClustering

def printTable(instances):
    for instance in instances:
        if instance != None:
            line = instance[0] + "\t"
            for i in range(1, len(instance)):
                line += "%.2f " % instance[i]
            print(line)


######################################################################
# Test code
######################################################################

dataset = loadCSV("iris.csv")
label_dict = {
    "Iris-setosa": 1,
    "Iris-versicolor": 2,
    "Iris-virginica": 3
}

# Normalizing data
train_vals = normalize([list(datum)[1:] for datum in dataset], axis=0).tolist()
for i in range(0, len(dataset)):
    train_vals[i].insert(0, dataset[i][0])
target_labels = [label_dict[instance[0]] for instance in dataset]

dist_array = ["Euclidean", "Cosine", "Jaccard"]
dist_metric = dist_array[0]
while_conditions = ["CentriodPosition", "MinSSE", "MaxIter"]
while_cond = while_conditions[0]

for dist in dist_array:
    dist_metric = dist
    # clustering = kmeans(dataset, 3)
    print("----------\nDistance metric {}:".format(dist))
    tic = time.perf_counter()
    clustering = kmeans(train_vals, 3)
    toc = time.perf_counter()
    print(f"Completed in {toc - tic:0.4f} seconds")
    # print(f"WithinSS: {clustering['withinss']}")
    print(f"Iterations: {clustering['iterations']}")


    Code to convert clusters to predicted labels using the most frequent value in each cluster as the label
    pred_clusters = [[label_dict[instance[0]] for instance in cluster] for cluster in clustering['clusters']]
    pred_labels = []
    for cluster in pred_clusters:
        mode_val = mode(cluster, axis=0).mode
        mode_list = [mode_val[0] if len(mode_val) > 0 else 0] * len(cluster)
        pred_labels.extend(mode_list)
    
    # Calculate v_measure_score to get an idea of clustering
    score  = v_measure_score(target_labels, pred_labels)
    print(f"Accuracy: {score}")

    Calculaute iterations needed for each stop condition
    iter_centroid = clustering['iterations']
    iter_max_iter = kmeans_maxIter(train_vals, 3)['iterations']
    iter_min_sse = kmeans_minSS(train_vals, 3)['iterations']
    
    print("\nCentroid Iterations: {}, \nMax Iterations (100) {}, \nMinimum SS Iterations: {}".format(iter_centroid, iter_max_iter, iter_min_sse))