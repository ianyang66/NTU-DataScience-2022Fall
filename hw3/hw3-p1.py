import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier

# ===========================================================
#                          Load Data
# ===========================================================
indexes = pd.read_csv('hw3_Data1/index.txt', delimiter = '\t', header = None).T
x = pd.read_csv('hw3_Data1/gene.txt', delimiter = ' ', header = None).to_numpy().T
y = pd.read_csv('hw3_Data1/label.txt', header = None).to_numpy()
y = (y>0).astype(int).reshape(y.shape[0])
index = indexes.iloc[0,].str.strip()
# print(y)
# print(np.shape(x))
# print(index)

# ===========================================================
#                          Load Data
# ===========================================================


# ===========================================================
#                       Feature ranking
# =========================================================== 
# TODO: Design your score function for feature selection
# TODO: To use the provided evaluation sample code, you need to generate ranking_idx, which is the sorted index of feature
def get_feature(score_func, x, y):
    select = SelectKBest(score_func=score_func, k=600)
    z = select.fit_transform(x,y)
    # print(z)
    print("After selecting best 600 features:", np.shape(z)) 
    filter = select.get_support()
    # print(filter)
    return index[filter], filter

def get_ranking_idx(features):
    ranking_idx = []
    for i in range(2000):
        if features[i] == True:
            ranking_idx.append(i)
    return ranking_idx
f, filter = get_feature(chi2, x, y)
ranking_idx = get_ranking_idx(filter)
fo = index[ranking_idx]
fo1 = fo.tolist()
print(f"Number of features: {len(ranking_idx)}")
print("Select top 600 Feature:")
print(fo1)



# ===========================================================
#                       Feature ranking
# =========================================================== 


# ===========================================================
#                       Feature evaluation
# =========================================================== 
# Use a simple dicision tree with 5-fold validation to evaluate the feature selection result.
# You can try other classifier and hyperparameter.
score_history = []
for m in range(5, 2001, 5):
    # Select Top m feature
    x_subset = x[:, ranking_idx[:m]]

    # Build random forest
    clf = DecisionTreeClassifier(random_state=0)
    # clf = SVC(kernel='rbf', random_state=0) #build SVM
    # clf = RandomForestClassifier(random_state=0) #build RandomForest

    # Calculate validation score
    scores = cross_val_score(clf, x_subset, y, cv=5)

    # Save the score calculated with m feature
    score_history.append(scores.mean())

# Report best accuracy.
num_feature = np.argmax(score_history)*5+5
f_new = index[ranking_idx[:num_feature]]
f_new_l = f_new.tolist()

# print("ranking_idx:")
# print(ranking_idx.tolist())
print(f"Max of Decision Tree: {max(score_history)}")
# print(f"Max of SVM: {max(score_history)}")
# print(f"Max of Radom Forest: {max(score_history)}")
print(f"Number of features: {num_feature}")
# print("Feature:")
#　print(f_new_l)

# ===========================================================
#                       Feature evaluation
# =========================================================== 


# ===========================================================
#                       Visualization
# =========================================================== 
plt.plot(range(5, 2001, 5), score_history, c='blue')
plt.title('Original')
plt.xlabel('Number of features')
plt.ylabel('Cross-validation score')
plt.legend(['Decision Tree'])
# plt.legend(['SVM'])
# plt.legend(['Ramdom Forest'])
plt.savefig('3-1_result.png')

# ===========================================================
#                       Visualization
# =========================================================== 

