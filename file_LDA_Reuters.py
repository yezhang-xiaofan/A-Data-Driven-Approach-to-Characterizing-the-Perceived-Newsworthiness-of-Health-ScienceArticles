__author__ = 'zhangye'
from factiva_model import process_file
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
def get_X_y():
    X,y,interest = process_file("reuters/all_reuters_article_info.csv","reuters/all_reuters_matched_articles_filtered.csv")
    vectorizer = CountVectorizer(stop_words="english",
                                    min_df=2,
                                    token_pattern=r"(?u)95% confidence interval|95% CI|95% ci|[a-zA-Z0-9_*\-][a-zA-Z0-9_/*\-]+",
                                    binary=False, max_features=50000)
    X = vectorizer.fit_transform(X)
    return X,np.array(y),vectorizer
if __name__ == "__main__":
    root = "reuters/"
    out_data = open(root+"LDA_data",'wb')
    out_label = open(root+"LDA_label",'wb')
    X, y, vectorizer = get_X_y()
    #X = X.toarray()
    indices = X.indices
    indptr = X.indptr
    data = X.data
    for i in range(X.shape[0]):
        index = indices[indptr[i]:indptr[i+1]]
        datas = data[indptr[i]:indptr[i+1]]
        out_data.write(str(len(index))+" ")
        for j in range(len(index)):
            out_data.write(str(index[j])+":"+str(datas[j])+" ")
        out_data.write("\n")
        out_label.write(str(1 if y[i]==1 else 0)+"\n")
    out_data.close()
    out_label.close()