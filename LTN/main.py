
# coding: utf-8

# In[1]:


#coding:utf8
import matplotlib
import os
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
else:
    pass
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy
from utils import Predicate,Clause,KnowledgeBase, Propositional
from utils import load_knowledge_base,load_propositional
from models import LTN
import pickle
import numpy as np
import seaborn as sns
sns.set(style="white", context="talk")


# In[2]:


def get_accuracy(model,kb):
    results=[]
    for clause in kb.clauses:
        o1,o2=model.forward(clause)
        if o2.data.numpy()[0][0]>0.9:
            results.append(1.0)
        else:
            results.append(0.0)

    return sum(results)/len(kb.clauses)


# In[3]:


def test_model(model,kb1, kb2,filename):
    kb_train=kb1.union(kb2)
    optimizor=torch.optim.Adam(model.parameters(),lr=0.001)
    mone=torch.FloatTensor([-1])
    one=torch.FloatTensor([1])
    average_prob=[]
    averate_loss=[]
    best_accuracy1=0.0
    best_accuracy2=0.0
    for i in tqdm(range(1000)):
        optimizor.zero_grad()
        total_probability=0.0
        total_loss=0.0
        for clause in kb_train.clauses:
            loss,prob=model.forward(clause=clause)
            loss.backward(one)
            total_probability+=prob.data.numpy()[0]
            total_loss+=loss.data.numpy()[0]
        optimizor.step()
        average_prob.append(total_probability/len(kb_train.clauses))
        averate_loss.append(total_loss/len(kb_train.clauses))
        accuracy1=get_accuracy(model,kb1)
        accuracy2=get_accuracy(model,kb2)
        if accuracy1+accuracy2>best_accuracy1+best_accuracy2:
            best_accuracy1=accuracy1
            best_accuracy2=accuracy2
    pickle.dump((average_prob,averate_loss,best_accuracy1,best_accuracy2), open("./results/%s"%filename, "wb" ))


# In[4]:


kb1=load_knowledge_base('./facts1.txt')
kb2=load_knowledge_base('./facts2.txt')
propositionals=load_propositional('./knowledge.txt')
gkbs1=[]
for p in propositionals:
    gkbs1.append(p.generate_knowledge_base('abcdefgh',change_weight=False))
gkb1=gkbs1[0]
for tkb in gkbs1[1:]:
    gkb1=gkb1.union(tkb)
gkbs2=[]
for p in propositionals:
    gkbs2.append(p.generate_knowledge_base('ijklmn',change_weight=False))
gkb2=gkbs2[0]
for tkb in gkbs2[1:]:
    gkb2=gkb2.union(tkb)

gkbs3=[]
for p in propositionals:
    gkbs3.append(p.generate_knowledge_base('abcdefgh',change_weight=True))
gkb3=gkbs3[0]
for tkb in gkbs3[1:]:
    gkb3=gkb3.union(tkb)
gkbs4=[]
for p in propositionals:
    gkbs4.append(p.generate_knowledge_base('ijklmn',change_weight=True))
gkb4=gkbs4[0]
for tkb in gkbs4[1:]:
    gkb4=gkb4.union(tkb)


# In[5]:


emb_dim=50


# In[6]:


emb_dim_range=list(range(10,20,5))+list(range(20,101,20))


# In[ ]:


for emb_dim in emb_dim_range:
    test_model(
        model=LTN(emb_dim,'abcdefghijklmn',[['S',1],['F',2],['C',1]], CLTN=True),
        kb1=kb1.union(gkb3),
        kb2=kb2.union(gkb4),
        filename='LTN_Learn_emb_dim=%d.pkl'%(emb_dim)
    )


# In[80]:


accuracys1=[]
accuracys2=[]
for emb_dim in emb_dim_range:
    prob,loss,first,second=pickle.load(open('./results/LTN_Learn_emb_dim=%d.pkl'%(emb_dim),'rb'))
    accuracys1.append(first)
    accuracys2.append(second)
plt.plot(emb_dim_range,accuracys1,label='Group1')
plt.plot(emb_dim_range,accuracys2,label='Group2')
plt.legend()
plt.xlabel('Vector Length')
plt.ylabel('Accuracy')
plt.savefig('./Report/img/curve4.pdf')
plt.show()

