import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy
from utils import Predicate, Clause, KnowledgeBase, Propositional
from utils import load_knowledge_base, load_propositional


class GConstants(nn.Module):
    def __init__(self, constants, emb_dim):
        super(GConstants, self).__init__()
        self.symbol2id = dict()
        self.id2symbol = dict()
        for i, s in enumerate(constants):
            self.symbol2id[s] = i
            self.id2symbol[i] = s
        self.embeddings = nn.Embedding(len(self.symbol2id), emb_dim)

    def forward(self, constants):
        constant_id = []
        for c in constants:
            constant_id.append(self.symbol2id[c])
        embs = self.embeddings(Variable(torch.LongTensor(constant_id)))
        return embs


class LTN_GPredicate(nn.Module):
    def __init__(self, name, variable_count, emb_dim):
        super(LTN_GPredicate, self).__init__()
        self.name = name
        self.variable_count = variable_count
        self.emb_dim = emb_dim
        m = variable_count * emb_dim
        self.variable_count = variable_count
        self.bilinear = nn.Bilinear(m, m, emb_dim, bias=False)
        self.linear1 = nn.Linear(m, emb_dim, bias=True)
        self.linear2 = nn.Linear(emb_dim, 1, bias=False)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Sigmoid()

    def forward(self, embs, negation):
        embs = torch.cat(embs).view(1, -1)
        output = self.bilinear(embs, embs) + self.linear1(embs)
        output = self.activation1(output)
        output = self.linear2(output)
        output = self.activation2(output)
        if negation:
            output = 1.0 - output
        return output


class CLTN_GPredicate(nn.Module):
    def __init__(self, name, variable_count, emb_dim):
        super(CLTN_GPredicate, self).__init__()
        self.name = name
        self.variable_count = variable_count
        self.emb_dim = emb_dim
        m = variable_count * emb_dim
        self.variable_count = variable_count

        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(16 * (emb_dim // 8)**2, 1), nn.Sigmoid())

    def forward(self, embs, negation):
        embs = embs.t().mm(embs)
        embs = embs.view(1, 1, embs.size()[0], embs.size()[1])
        conv = self.conv(embs)
        conv = conv.view(conv.size()[0], -1)
        output = self.linear(conv)
        if negation:
            output = 1.0 - output
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LTN(nn.Module):
    def __init__(self, emb_dim, constants, predicates, CLTN):
        super(LTN, self).__init__()
        self.emb_dim = emb_dim
        self.Constants = GConstants(constants, emb_dim)
        self.Predicates = dict()
        for name, n in predicates:
            if CLTN == False:
                self.Predicates[name] = LTN_GPredicate(name, n, emb_dim)
            else:
                self.Predicates[name] = CLTN_GPredicate(name, n, emb_dim)
                self.Predicates[name].apply(weights_init)

    def forward(self, clause):
        Phi = None
        v = Variable(torch.FloatTensor([clause.v]))
        w = Variable(torch.FloatTensor([clause.w]))
        for predicate in clause.predicates:
            negation = predicate.negation
            name = predicate.name
            constants = predicate.variables
            embs = self.Constants.forward(constants)
            output = self.Predicates[name].forward(embs, negation)
            if Phi is None or Phi.data.numpy()[0] < output.data.numpy()[0]:
                Phi = output
        if (Phi > w).data.all():
            loss = Phi - w
        elif (Phi < v).data.all():
            loss = v - Phi
        else:
            loss = Variable(torch.FloatTensor([0]), requires_grad=True)
        loss = loss * clause.weight
        return loss, Phi

    def parameters(self):
        results = list(self.Constants.parameters())
        for name in self.Predicates:
            results += list(self.Predicates[name].parameters())
        return results
