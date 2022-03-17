# TODO: Add fit and sample methods

import numpy as np
from mbi import FactoredInference, Dataset, Domain
from scipy import sparse
from disjoint_set import DisjointSet
import networkx as nx
import itertools
from scipy.special import logsumexp
import argparse
import json

"""
WRAPPER for MST synthesizer from Private PGM:
https://github.com/ryan112358/private-pgm/tree/e9ea5fcac62e2c5b92ae97f7afe2648c04432564

This is a generalization of the winning mechanism from the 
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.  
"""

class MSTSynthesizer():
    """
    
    dataset:
    """

    Domains = {
        "fake": "fake-domain.json",
        "compas": "compas-domain.json"
    }
    
    def __init__(self,
                 domain="compas",
                 epsilon=0.1,
                 delta=1e-9,
                 degree=2,
                 num_marginals=None,
                 max_cells=10000,
                 domain_path=None,
                 seed=42,
                 custom_cliques=False,
                 cliques_set=[]
                ):
        
        if domain_path is None:
            domain_name = self.Domains[domain]
            with open(domain_name) as json_file:
                dict_domain = json.load(json_file)
        else:
            with open(domain_path) as json_file:
                dict_domain = json.load(json_file)
                
        if dict_domain is None:
            raise ValueError("Domain file not found for: " + domain + " and " + domain_name)
        self.domain = Domain.fromdict(dict_domain)
        self.degree = degree
        self.epsilon = epsilon
        self.delta = delta
        self.num_marginals = num_marginals
        self.max_cells = max_cells
        self.seed = seed

        self.synthesizer = None
        self.num_rows = None

        # For Homework: allow custom clique sets
        self.custom_cliques = custom_cliques
        self.cliques_set = cliques_set

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.num_rows = len(data)
        print(self.domain)
        print(data.columns)
        prng = np.random.RandomState(self.seed)
        self.data = Dataset(df=data, domain=self.domain)

        workload = list(itertools.combinations(self.data.domain, self.degree))
        workload = [cl for cl in workload if self.data.domain.size(cl) <= self.max_cells]
        if self.num_marginals is not None:
            workload = [workload[i] for i in prng.choice(len(workload), self.num_marginals, replace=False)]

        self.MST(self.data, self.epsilon, self.delta)
    
    def sample(self, samples=None):
        if samples is None:
            samples = self.num_rows
        data = self.synthesizer.synthetic_data(rows=samples)
        decompressed = self.undo_compress_fn(data)
        return decompressed.df

    def MST(self, data, epsilon, delta):
        rho = cdp_rho(epsilon, delta)
        sigma = np.sqrt(3/(2*rho))
        cliques = [(col,) for col in data.domain]
        log1 = self.measure(data, cliques, sigma)
        data, log1, undo_compress_fn = self.compress_domain(data, log1)

        # Here's the decompress function
        self.undo_compress_fn = undo_compress_fn

        if self.custom_cliques:
            cliques = self.cliques_set
        else:
            cliques = self.select(data, rho/3.0, log1)
        print(cliques)
        log2 = self.measure(data, cliques, sigma)
        engine = FactoredInference(data.domain, iters=5000)
        est = engine.estimate(log1+log2)

        # Here's the synthesizer
        self.synthesizer = est

        # synth = est.synthetic_data()
        # return undo_compress_fn(synth)

    def measure(self, data, cliques, sigma, weights=None):
        if weights is None:
            weights = np.ones(len(cliques))
        weights = np.array(weights) / np.linalg.norm(weights)
        measurements = []
        for proj, wgt in zip(cliques, weights):
            x = data.project(proj).datavector()
            y = x + np.random.normal(loc=0, scale=sigma/wgt, size=x.size)
            Q = sparse.eye(x.size)
            measurements.append( (Q, y, sigma/wgt, proj) )
        return measurements

    def compress_domain(self, data, measurements):
        supports = {}
        new_measurements = []
        for Q, y, sigma, proj in measurements:
            col = proj[0]
            sup = y >= 3*sigma
            supports[col] = sup
            if supports[col].sum() == y.size:
                new_measurements.append( (Q, y, sigma, proj) )
            else: # need to re-express measurement over the new domain
                y2 = np.append(y[sup], y[~sup].sum())
                I2 = np.ones(y2.size)
                I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
                y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
                I2 = sparse.diags(I2)
                new_measurements.append( (I2, y2, sigma, proj) )
        undo_compress_fn = lambda data: self.reverse_data(data, supports)
        return self.transform_data(data, supports), new_measurements, undo_compress_fn

    def exponential_mechanism(self, q, eps, sensitivity, prng=np.random, monotonic=False):
        coef = 1.0 if monotonic else 0.5
        scores = coef*eps/sensitivity*q
        probas = np.exp(scores - logsumexp(scores))
        return prng.choice(q.size, p=probas)

    def select(self, data, rho, measurement_log, cliques=[]):
        engine = FactoredInference(data.domain, iters=1000)
        est = engine.estimate(measurement_log)

        weights = {}
        candidates = list(itertools.combinations(data.domain.attrs, 2))
        for a, b in candidates:
            xhat = est.project([a, b]).datavector()
            x = data.project([a, b]).datavector()
            weights[a,b] = np.linalg.norm(x - xhat, 1)

        self.T = nx.Graph()
        self.T.add_nodes_from(data.domain.attrs)
        ds = DisjointSet()

        for e in cliques:
            self.T.add_edge(*e)
            ds.union(*e)

        r = len(list(nx.connected_components(self.T)))
        epsilon = np.sqrt(8*rho/(r-1))
        for i in range(r-1):
            candidates = [e for e in candidates if not ds.connected(*e)]
            wgts = np.array([weights[e] for e in candidates])
            idx = self.exponential_mechanism(wgts, epsilon, sensitivity=1.0)
            e = candidates[idx]
            self.T.add_edge(*e)
            ds.union(*e)

        return list(self.T.edges)

    def transform_data(self, data, supports):
        df = data.df.copy()
        newdom = {}
        for col in data.domain:
            support = supports[col]
            size = support.sum()
            newdom[col] = int(size)
            if size < support.size:
                newdom[col] += 1
            mapping = {}
            idx = 0
            for i in range(support.size):
                mapping[i] = size
                if support[i]:
                    mapping[i] = idx
                    idx += 1
            assert idx == size
            df[col] = df[col].map(mapping)
        newdom = Domain.fromdict(newdom)
        return Dataset(df, newdom)

    def reverse_data(self, data, supports):
        df = data.df.copy()
        newdom = {}
        for col in data.domain:
            support = supports[col]
            mx = support.sum()
            newdom[col] = int(support.size)
            idx, extra = np.where(support)[0], np.where(~support)[0]
            mask = df[col] == mx
            if extra.size == 0:
                pass
            else:
                df.loc[mask, col] = np.random.choice(extra, mask.sum())
            df.loc[~mask, col] = idx[df.loc[~mask, col]]
        newdom = Domain.fromdict(newdom)
        return Dataset(df, newdom)

    def display_MST_graph(self):
        nx.draw(self.T, with_labels = True)

# TODO: Get rid of this. Here now for convenience.

"""
   Copyright 2020 (https://github.com/IBM/discrete-gaussian-differential-privacy)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

#Code for computing approximate differential privacy guarantees
# for discrete Gaussian and, more generally, concentrated DP
# See https://arxiv.org/abs/2004.00010
# - Thomas Steinke dgauss@thomas-steinke.net 2020

import math
import matplotlib.pyplot as plt


#*********************************************************************
#Now we move on to concentrated DP
    
#compute delta such that
#rho-CDP implies (eps,delta)-DP
#Note that adding cts or discrete N(0,sigma2) to sens-1 gives rho=1/(2*sigma2)

#start with standard P[privloss>eps] bound via markov
def cdp_delta_standard(rho,eps):
    assert rho>=0
    assert eps>=0
    if rho==0: return 0 #degenerate case
    #https://arxiv.org/pdf/1605.02065.pdf#page=15
    return math.exp(-((eps-rho)**2)/(4*rho))

#Our new bound:
# https://arxiv.org/pdf/2004.00010v3.pdf#page=13
def cdp_delta(rho,eps):
    assert rho>=0
    assert eps>=0
    if rho==0: return 0 #degenerate case

    #search for best alpha
    #Note that any alpha in (1,infty) yields a valid upper bound on delta
    # Thus if this search is slightly "incorrect" it will only result in larger delta (still valid)
    # This code has two "hacks".
    # First the binary search is run for a pre-specificed length.
    # 1000 iterations should be sufficient to converge to a good solution.
    # Second we set a minimum value of alpha to avoid numerical stability issues.
    # Note that the optimal alpha is at least (1+eps/rho)/2. Thus we only hit this constraint
    # when eps<=rho or close to it. This is not an interesting parameter regime, as you will
    # inherently get large delta in this regime.
    amin=1.01 #don't let alpha be too small, due to numerical stability
    amax=(eps+1)/(2*rho)+2
    for i in range(1000): #should be enough iterations
        alpha=(amin+amax)/2
        derivative = (2*alpha-1)*rho-eps+math.log1p(-1.0/alpha)
        if derivative<0:
            amin=alpha
        else:
            amax=alpha
    #now calculate delta
    delta = math.exp((alpha-1)*(alpha*rho-eps)+alpha*math.log1p(-1/alpha)) / (alpha-1.0)
    return min(delta,1.0) #delta<=1 always

#Above we compute delta given rho and eps, now we compute eps instead
#That is we wish to compute the smallest eps such that rho-CDP implies (eps,delta)-DP
def cdp_eps(rho,delta):
    assert rho>=0
    assert delta>0
    if delta>=1 or rho==0: return 0.0 #if delta>=1 or rho=0 then anything goes
    epsmin=0.0 #maintain cdp_delta(rho,eps)>=delta
    epsmax=rho+2*math.sqrt(rho*math.log(1/delta)) #maintain cdp_delta(rho,eps)<=delta
    #to compute epsmax we use the standard bound
    for i in range(1000):
        eps=(epsmin+epsmax)/2
        if cdp_delta(rho,eps)<=delta:
            epsmax=eps
        else:
            epsmin=eps
    return epsmax

#Now we compute rho
#Given (eps,delta) find the smallest rho such that rho-CDP implies (eps,delta)-DP
def cdp_rho(eps,delta):
    assert eps>=0
    assert delta>0
    if delta>=1: return 0.0 #if delta>=1 anything goes
    rhomin=0.0 #maintain cdp_delta(rho,eps)<=delta
    rhomax=eps+1 #maintain cdp_delta(rhomax,eps)>delta
    for i in range(1000):
        rho=(rhomin+rhomax)/2
        if cdp_delta(rho,eps)<=delta:
            rhomin=rho
        else:
            rhomax=rho
    return rhomin


