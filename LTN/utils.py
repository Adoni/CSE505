import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy


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

    def generate_knowledge_base(self, constants):
        self.knowledge_base = KnowledgeBase([self.clause])
        for condition in self.conditions:
            new_knowledge_base = self.bound_knowledge_base(
                self.knowledge_base, condition[1], constants, condition[0])
            self.knowledge_base = new_knowledge_base
        return self.knowledge_base

    def bound_knowledge_base(self, knowledge_base, variable, constants,
                             existential):
        clauses = []
        for clause in knowledge_base.clauses:
            clauses += self.bound_clause_variable_with_constants(
                clause, variable, constants, existential)
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
