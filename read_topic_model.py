__author__ = 'zhangye'
import predictPR
import predictNConPR
import numpy as np
import file_for_LDA
import file_LDA_Reuters
import file_LDA_JAMA
import sys
#X, y, vectorizer = file_for_LDA.get_X_y_NC()
#X,y, vectorizer = file_LDA_Reuters.get_X_y()
#X, y, vectorizer = file_for_LDA.get_X_y_PR()
X,y,vectorizer = file_LDA_JAMA.get_X_y()
topic_model = open('../slda/jama_20/final.model.text')
alpha = float(topic_model.readline().split(":")[1].strip())
num_topic = int(topic_model.readline().split(":")[1].strip())
size_voc = int(topic_model.readline().split(":")[1].strip())
num_class = int(topic_model.readline().split(":")[1].strip())
topic_model.readline()
word_dist = np.zeros((num_topic,size_voc))
for n in range(num_topic):
    dist = np.array(map(lambda x:float(x), topic_model.readline().strip().split(" ")))
    word_dist[n] = dist
topic_model.readline()
etas = map(lambda x: float(x),topic_model.readline().strip().split(" "))
#display top words for each topic
sorted_topics = sorted(zip(etas,range(len(etas))))
for s in range(len(sorted_topics)):
    print ("coefficient: "+str(-sorted_topics[s][0])+"\n")
    index = sorted_topics[s][1]
    sorted_word = sorted(zip(word_dist[index],range(size_voc)),reverse=True)
    for j in range(30):
        sys.stdout.write(vectorizer.get_feature_names()[(sorted_word[j][1])]+" ")
    print("\n")