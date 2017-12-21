#coding:utf8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy
import pandas as pd
import numpy


class Predicate:
    def __init__(self, name, variables, negation):
        self.name = name
        self.variables = variables
        self.negation = negation

    def show(self):
        s = ''
        if self.negation:
            s += '¬'
        s = '%s%s(%s)' % (s, self.name, ', '.join(self.variables))
        return s


class Clause:
    def __init__(self, v, w, predicates, weight=1.0):
        self.v = float(v)
        self.w = float(w)
        self.predicates = predicates
        self.weight = weight

    def show(self):
        ss = []
        for p in self.predicates:
            ss.append(p.show())
        ss = ' ∨ '.join(ss)
        #         ss='%0.2f: '%(self.weight)+ss
        return ss


class KnowledgeBase:
    def __init__(self, clauses):
        self.clauses = clauses

    def show(self):
        for i, c in enumerate(self.clauses):
            print(c.show())

    def union(self, knowledge_base):
        new_knowledge_base = copy.deepcopy(self)
        for clause in knowledge_base.clauses:
            new_knowledge_base.clauses.append(clause)
        return new_knowledge_base


class Propositional:
    def __init__(self, v, w, conditions, clause):
        self.v = v
        self.w = w
        self.conditions = conditions
        self.clause = clause

    def generate_knowledge_base(self, constants, change_weight):
        self.knowledge_base = KnowledgeBase([self.clause])
        for condition in self.conditions:
            new_knowledge_base = self.bound_knowledge_base(
                self.knowledge_base, condition[1], constants, condition[0],
                change_weight)
            self.knowledge_base = new_knowledge_base
        return self.knowledge_base

    def bound_knowledge_base(self, knowledge_base, variable, constants,
                             existential, change_weight):
        clauses = []
        for clause in knowledge_base.clauses:
            clauses += self.bound_clause_variable_with_constants(
                clause, variable, constants, existential)
        if change_weight:
            for i in range(len(clauses)):
                clauses[i].weight = 1.0 / len(clauses)
        return KnowledgeBase(clauses=clauses)

    def bound_clause_variable_with_constants(self, clause, variable, constants,
                                             existential):
        clauses = []
        for constant in constants:
            clauses.append(
                self.bound_clause_variable_with_constant(
                    clause, variable, constant))
        if existential:
            all_predicates = []
            for clause in clauses:
                all_predicates += clause.predicates
            return [Clause(clauses[0].v, clauses[0].w, all_predicates)]
        else:
            return clauses

    def bound_clause_variable_with_constant(self, clause, variable, constant):
        new_clause = copy.deepcopy(clause)
        for i in range(len(clause.predicates)):
            for j in range(len(new_clause.predicates[i].variables)):
                if new_clause.predicates[i].variables[j] == variable:
                    new_clause.predicates[i].variables[j] = constant
        return new_clause


def load_knowledge_base(filename):
    clauses = []
    for line in open(filename):
        line = line.strip().split('|')
        v, w = line[0].split(',')
        line = line[1].strip().split(',')
        if line[0] == 'not':
            negation = True
            line = line[1:]
        else:
            negation = False
        predicate = Predicate(
            name=line[0], negation=negation, variables=line[1:])
        clauses.append(Clause(v=v, w=w, predicates=[predicate]))
    return KnowledgeBase(clauses)


def load_propositional(filename):
    propositionals = []
    for line in open(filename):
        line = line.strip().split('|')
        condition = []
        v, w = line[0].split(',')
        for c in line[1].strip().split(','):
            c = c.strip().split(' ')
            if c[0] == 'all':
                c[0] = False
            else:
                c[0] = True
            condition.append(c)
        predicates = []
        for predicate in line[2:]:
            predicate = predicate.strip().split(',')
            if predicate[0] == 'not':
                predicate = Predicate(
                    name=predicate[1], negation=True, variables=predicate[2:])
            else:
                predicate = Predicate(
                    name=predicate[0], negation=False, variables=predicate[1:])
            predicates.append(predicate)
        propositionals.append(
            Propositional(v, w, condition, Clause(v, w, predicates)))
    return propositionals


def get_DF_S(model, constants):
    constants = list(constants)
    df = pd.DataFrame(index=constants, columns=['S'])
    for a in constants:
        clause = Clause(1, 1,
                        [Predicate(name='S', variables=[a], negation=False)])
        result = model.forward(clause)
        df['S'][a] = '%0.2f' % result[1].data.numpy()[0]
    return df


def get_DF_C(model, constants):
    constants = list(constants)
    df = pd.DataFrame(index=constants, columns=['C'])
    for a in constants:
        clause = Clause(1, 1,
                        [Predicate(name='C', variables=[a], negation=False)])
        result = model.forward(clause)
        df['C'][a] = '%0.2f' % result[1].data.numpy()[0]
    return df


def get_DF_F(model, constants):
    constants = list(constants)
    df = pd.DataFrame(index=constants, columns=constants)
    for a in constants:
        for b in constants:
            #             if a>=b:
            #                 df[b][a]='-'
            #                 continue
            clause = Clause(
                1, 1, [Predicate(name='F', variables=[a, b], negation=False)])
            result = model.forward(clause)
            df[b][a] = '%0.2f' % result[1].data.numpy()[0]
    return df


def get_DF(model, constants):
    df1 = get_DF_S(model, constants)
    df2 = get_DF_C(model, constants)
    df3 = get_DF_F(model, constants)
    df = pd.concat([df1, df2, df3], axis=1)
    return df


def get_accuracy(model, kb):
    results = []
    for clause in kb.clauses:
        o1, o2 = model.forward(clause)
        if o2.data.numpy()[0][0] > 0.9:
            results.append(1.0)
        else:
            results.append(0.0)

    return sum(results) / len(kb.clauses)


def show_learned_propositionals(model, propositionals):
    results = pd.DataFrame(
        index=range(len(propositionals)),
        columns=['Propositional', 'Group1', 'Group2'])
    for i, propositional in enumerate(propositionals):
        total = 0
        true_count = 0
        kkk1 = propositional.generate_knowledge_base(
            'abcdefgh', change_weight=False)
        kkk2 = propositional.generate_knowledge_base(
            'ijklmn', change_weight=False)
        a1 = get_accuracy(model, kkk1)
        a2 = get_accuracy(model, kkk2)
        results.iloc[i] = dict(
            Propositional=propositional.clause.show(), Group1=a1, Group2=a2)
    return results
