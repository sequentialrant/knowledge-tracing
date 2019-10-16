"""
Alef Education
Author: Srinidhi Havaldar
Date: January 08, 2019
"""

import csv
import numpy as np
from scipy.optimize import minimize
from dataloader import DataLoader


class BKT(DataLoader):
    """
    A gradient descent implementation of the Bayesian Knowledge Tracing algorithm as described in the paper:
    "Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge. Corbett & Anderson (1995)"

    Args:
        DataLoader (object)
        data (list): student transactional data in the increasing order of timestamp

    Return:
        skill_models (dict): a dictionary of model parameters for every skill
        written_model_parameters (csv): writes to a CSV file, student-specific skill mastery, guess, slip,
        and transition parameters
    """
    def __init__(self, data_loader, data):
        super().__init__()
        self.data = data
        self.header = [x.lower() for x in data[0]]
        del data[0]

        self.guess_bound = data_loader.guess_bnd
        self.slip_bound = data_loader.slip_bnd
        self.method = data_loader.solver
        self.tol = data_loader.tolerance
        self.max_iters = data_loader.m_iters

        self.student = self.header.index(data_loader.student)
        self.skill = self.header.index(data_loader.mlo_id)
        self.order = self.header.index(data_loader.start_time)
        self.correct = self.header.index(data_loader.q_score)
        self.skills = list(set([x[self.skill] for x in self.data]))

        self.new_mastery = None
        self.prev_mastery = None
        self.is_correct = None
        self.sk_params = None

        self.skill_models = {}
        self.param_file = data_loader.param_f

    def compute_likelihood(self):
        """
        Computes the likelihood of answering the next question correctly

        :return: Probability of answering the next question correct
        """
        return (self.prev_mastery * (1 - self.sk_params[2])) + \
               ((1 - self.prev_mastery) * self.sk_params[1])

    def compute_loss(self, predicted, eps=1e-15):
        """
        Computes logarithmic loss

        :param predicted:
        :param eps:
        :return: log-loss
        """
        p = np.clip(predicted, eps, 1 - eps)

        if self.is_correct == 1:
            return -np.log(p)
        else:
            return -np.log(1 - p)

    def update_mastery(self):
        """
        Computes updated mastery depending on student's previous mastery and correct/incorrect response to a question

        :return: updated mastery
        """
        guess = self.sk_params[1]
        slip = self.sk_params[2]
        trans = self.sk_params[3]

        correct_contrib = (self.prev_mastery * (1 - slip)) / (self.prev_mastery * (1 - slip) +
                                                              (1 - self.prev_mastery) * guess)
        incorrect_contrib = (self.prev_mastery * slip) / (self.prev_mastery * slip +
                                                          (1 - self.prev_mastery) * (1 - guess))
        updated_mastery = self.is_correct * correct_contrib + incorrect_contrib * (1 - self.is_correct)

        return updated_mastery + ((1 - updated_mastery) * trans)

    def objective(self, params, subset):
        """
        Computes error to minimize the log-loss

        :param params:
        :param subset:
        :return: error
        """
        self.sk_params = params
        st = subset[0][self.student]
        self.prev_mastery = self.sk_params[0]
        self.new_mastery = self.sk_params[0]
        error = 0

        for l in subset:
            self.is_correct = l[self.correct]
            if l[self.student] != st:
                self.prev_mastery = self.sk_params[0]
            error += self.compute_loss(self.compute_likelihood())
            self.new_mastery = self.update_mastery()
            st = l[self.student]
            self.prev_mastery = self.new_mastery
        return error

    def fit(self):
        """
        Fits Bayesian Knowledge Tracing using L-BFGS-B optimizer

        :return: best model parameters for each skill
        """
        bound = [(0.01, 1), (0.01, self.guess_bound), (0.01, self.slip_bound),
                 (0.01, 1)]  # to avoid model degeneracy

        for sk in self.skills:
            subset = [i for i in self.data if i[self.skill] == sk]
            self.sk_params = np.array([0.6, 0.2, 0.1, 0.3])
            model = minimize(self.objective, self.sk_params, args=(subset,),
                             method=self.method, bounds=bound, options={'maxiter': self.max_iters})
            best_model = [model.x, model.fun]
            # print(sk, best_model[0], best_model[1])
            self.skill_models[sk] = best_model
        return

    def predict(self, params=None):
        """
        Applies estimated parameters to compute skill-specific mastery for every student.

        If params=None, predict uses the parameters from the skill_models, otherwise parameters associated
        with every skill should be passed in the format dict[skill]: [mastery, guess, slip, transition]

        :params (dict):
        :return: CSV file
        """
        writer = csv.writer(open(self.param_file + "_model_params.csv", "w"))

        self.header.extend(
            ['initial_mastery',  'guess', 'slip', 'transition', 'new_mastery', 'likelihood'])
        writer.writerow(self.header)

        if params:
            self.sk_params = params[self.data[0][self.skill]]
        else:
            skl = self.data[0][self.skill]
            self.sk_params = self.skill_models[skl][0]
            
        self.prev_mastery = self.sk_params[0]
        st = self.data[0][self.student]
        sk = self.data[0][self.skill]

        for d in self.data:
            if sk != d[self.skill]:
                if params:
                    self.sk_params = params[d[self.skill]]
                else:
                    self.sk_params = self.skill_models[d[self.skill]][0]
            if st != d[self.student]:
                self.prev_mastery = self.sk_params[0]
            self.is_correct = d[self.correct]
            self.new_mastery = self.update_mastery()
            d.extend([x for x in self.sk_params])
            d.extend([self.new_mastery, self.compute_likelihood()])
            writer.writerow(d)
            self.prev_mastery = self.new_mastery
            st = d[self.student]
            sk = d[self.skill]
        return
