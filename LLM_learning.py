import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn. feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

newsgroups_train=fetch_20newsgroups(subset='train',categories=['comp.graphics','sci.space'])
newsgroups_test=fetch_20newsgroups(subset='test',categories=['comp.graphics','sci.space'])

vectorizer=CountVectorizer(stop_words='english',max_features=1000)
inputs_train=vectorizer.fit_transform(newsgroups_train.data).toarray()
inputs_test=vectorizer.transform(newsgroups_test.data).toarray()
inputs_train=torch.tensor(inputs_train,dtype=torch.float32)
labels_train=torch.tensor(newsgroups_train.target,dtype=torch.long)
inputs_test=torch.tensor(inputs_test,dtype=torch.long)
labels_test=torch.tensor(newsgroups_test.target,dtype=torch.long)
print(inputs_train,inputs_test)
print(newsgroups_train,newsgroups_test)

class net(nn.Module):
    def __init__(self):
        super(net.self).__init__()
        self.func1=nn.linear(16.32)
        self.func2=nn.linear(32,10)
    def forwards(self,x):
        x=torch.relu(self.func1(x))
        x=self.func2(x)
        return x