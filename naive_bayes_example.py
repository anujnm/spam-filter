
from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import sys


"""
Returns a dense vector indicative of the presence of features for a given email.
:param content: contents of the email that is to be processed
:rtype: Vector
"""
def getFeatures(content):
    # Hardcoding a feature set for now. 
    features = {'sunglasses': 0, 'cheap': 1, 'free': 2, 'money': 3,
                'time': 4, 'person': 5, 'kind': 6, 'linux': 7,
                'chinese': 8, 'peek': 9, 'curious': 10, 'charges': 11,
                'norco': 12, 'adderall': 13, 'limited': 14,
                'offer': 15, 'today': 16, 'price': 17, 'webcam': 18,
                'url': 19}

    result = [0 for _ in xrange(len(features))]

    for word in content.split():
        word = word.lower()
        if word in features:
            result[features[word]] = 1
    
    return Vectors.dense(result)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        raise Exception("Please provide a path to the labels file, and the emails folder")
    
    path_to_label_file = sys.argv[1]
    path_to_emails = sys.argv[2]

    sc = SparkContext(appName="spam-filter")

    # Process file containing data labels.
    # labels: {filename: isSpam (1 or 0)}
    labels = sc.textFile(path_to_label_file).\
        map(lambda line: line.split()).\
        map(lambda (num, name): (name, int(num))).\
        collectAsMap()

    # Parse email folder for all emails, convert into feature set
    # emails: [ (isSpam, content) ]
    emailFeatures = sc.wholeTextFiles(path_to_emails).\
        map(lambda (filename, content):
            (labels[filename.split('/')[-1]], content)).\
        map(lambda (isSpam, content):
            LabeledPoint(str(isSpam), getFeatures(content)))
    
    training, test = emailFeatures.randomSplit([0.6, 0.4], seed=9023)

    model = NaiveBayes.train(training, 1.0)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(
        lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(
        lambda (x, v): x == v).count() / test.count()
    print '\n'*20, 'ACCURACY', accuracy