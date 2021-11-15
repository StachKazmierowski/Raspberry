import math
import numpy as np
import pandas as pd
from solutions_evaluator import epsilon_value
import time
np.set_printoptions(suppress=True)
SIGNIFICANCE_CONST = 10**(-20)
from unsymmetrized_utils import payoff_matrix
import sys
EXPERIMENTS_RESULTS_PATH = "../tmp/"
from unsymmetrized_utils import get_rows, get_all_divides, add_column, get_col

def MWU_game_algorithm(A, B, n, fields_values, phi=1/2, steps_number=10000):
    row_strategies = get_all_divides(A, n)
    column_strategies = get_all_divides(B, n)
    rows_number = row_strategies.shape[0]
    print("ROWS_NUMBER:", rows_number)
    cols_number = column_strategies.shape[0]
    print("COLS_NUMBER:", cols_number)
    support_size = poly_size(rows_number)
    columns = np.zeros((0))
    j_played = np.array([])
    print("POLYSIZE: ", support_size)
    indexes = [i for i in range(poly_size(rows_number))]
    used_row_strategies = np.zeros((rows_number,))
    used_row_strategies[indexes] = 1
    p_t = np.ones((1, len(indexes)))
    p_t = normalize(p_t)
    j_sumed = np.zeros((cols_number, 1))
    smallest_column_payoff = 1
    p_best = p_t
    best_indexes = indexes
    start = time.time()
    k = 1
    tmp_mat = get_rows(row_strategies[indexes], column_strategies, fields_values)
    for i in range (1, steps_number):
        payoffs = np.matmul(p_t, tmp_mat)
        j_best_response = np.argmax(payoffs)
        if(payoffs[0, j_best_response] < smallest_column_payoff):
            smallest_column_payoff = payoffs[0, j_best_response]
            p_best = p_t
            best_indexes = indexes
        if not (j_played == j_best_response).any():
            column_index = (j_played < j_best_response).sum()
            column = get_col(row_strategies, column_strategies[j_best_response], fields_values)
            columns = add_column(columns, column, column_index)
            j_played = list(j_played)
            j_played.append(j_best_response)
            j_played = np.array(j_played)
        j_sumed[j_best_response] += 1
        m_t = tmp_mat[:,j_best_response]

        m_t_negative = (m_t < 0)
        p_t_significant = (p_t > SIGNIFICANCE_CONST)
        to_update = np.logical_or(m_t_negative, p_t_significant[0])
        m_t_updating = np.where(to_update,m_t,0)
        p_t_updating = np.where(to_update,p_t,0)
        p_t = np.multiply((1 - phi * m_t_updating), p_t_updating)

        j_distribution = j_sumed/j_sumed.sum()
        j_distribution_tmp = j_sumed[j_sumed > 0]/j_sumed.sum()
        if(i == k and i < steps_number/2):
            print(i)
            new_indexes = get_sorted_indexes(best_pure_responses_indexes(columns, j_distribution_tmp))
            p_t = update_p_t(p_t, indexes, new_indexes, 1/rows_number)
            indexes = new_indexes
            used_row_strategies[indexes] = 1
            k += 2**int(math.log2(i))
            tmp_mat = get_rows(row_strategies[indexes], column_strategies, fields_values)
        p_t = p_t/p_t.sum()
    j_distribution = j_sumed/ j_sumed.sum()
    p_best_big_support = translate_p_t(p_best, best_indexes, rows_number)
    j_distribution_tmp = j_sumed[j_sumed > 0] / j_sumed.sum()
    j_distribution_tmp = j_distribution_tmp.reshape(j_distribution_tmp.shape[0], 1)
    row_col_1 = (max(epsilon_value(p_best, j_distribution, tmp_mat)))
    row_col_2 = (max(epsilon_value(p_best_big_support, j_distribution_tmp, -columns)))
    row_col = max(row_col_1, row_col_2)
    duration = time.time() - start
    return duration, row_col, used_fraction_of_matrix(used_row_strategies.sum(), (j_sumed > 0).sum(), rows_number, cols_number)

def used_fraction_of_matrix(used_rows, used_cols, rows_num, cols_num):
    return (used_rows * cols_num + used_cols * rows_num - used_cols * used_rows) / (rows_num * cols_num)

def translate_p_t(p_t, indexes, rows_num):
    result = np.zeros((1, rows_num))
    for i in range(len(indexes)):
        result[0, indexes[i]] = p_t[0, i]
    return result

def update_p_t(p_t, old_indexes, new_indexes, update_value):
    p_t_new = np.zeros_like(p_t)
    for i in range(len(new_indexes)):
        if new_indexes[i] in old_indexes:
            index = old_indexes.index(new_indexes[i])
            p_t_new[0, i] = p_t[0, index]
    p_t_new[p_t_new == 0] = update_value
    return p_t_new

def get_sorted_indexes(indexes):
    list_indexes = list(indexes[:])
    list_indexes.sort()
    return list_indexes

def best_pure_responses_indexes(matrix, column_strat):
    row_payoffs = np.matmul(matrix, column_strat)
    indexes = np.argsort(row_payoffs, axis=0)
    return indexes[- poly_size(matrix.shape[0]):]

def worst_pure_responses_indexes(matrix, column_strat):
    row_payoffs = -np.matmul(matrix, column_strat)
    indexes = np.argsort(row_payoffs, axis=0)
    return indexes[ : poly_size(matrix.shape[0])]

def normalize(vector):
    return vector/vector.sum()

def start_vector(matrix, rows_number, columns_number):
    p_0 = np.zeros((1, rows_number))
    uniform = np.ones((columns_number, 1))
    uniform = normalize(uniform)
    best = worst_pure_responses_indexes(matrix, uniform)
    p_0[0][best] = 1
    p_0 = normalize(p_0)
    return p_0

def poly_size(number):
    return int(math.log(number) ** 2)

def save_experiments_results(data, name, folder=EXPERIMENTS_RESULTS_PATH):
    data.to_csv(folder + name)

def gen_name(A, B, n):
    return "(" + str(A) + "," + str(B) + "," + str(n) + ").csv"

values = np.array([1,2,3,4,5, 6,7])
print(MWU_game_algorithm(10,10,7,values, 1/128, 20000))