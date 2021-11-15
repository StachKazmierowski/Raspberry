import math
import numpy as np
import pandas as pd
from solutions_evaluator import possible_payoff_increase_B
SIGNIFICANCE_CONST = 10**(-20)
from mwu_without_time_hoop import MWU_game_algorithm

def next_bit_divide(divide):
    length = divide.shape[0]
    return_divide = np.array(divide)
    last_one_index = find_last_one_index(divide)
    if(last_one_index < length - 1):
        return_divide[last_one_index] = 0
        return_divide[last_one_index + 1] = 1
        return return_divide
    else:
        last_zero_index = find_last_zero_index(divide)
        # print("LAST ZERO INDEX:" , last_zero_index)
        number_of_ones_at_end = length - 1 - last_zero_index
        tmp_divide = next_bit_divide(divide[:last_zero_index + 1])
        # print("TMP divide: ", tmp_divide)
        return_divide = np.zeros((length)).astype(int)
        return_divide[:last_zero_index + 1] = tmp_divide[:]
        tmp_last_one_index = find_last_one_index(return_divide)
        # print(return_divide)
        return_divide[tmp_last_one_index + 1 : tmp_last_one_index + number_of_ones_at_end + 1] = 1
        return return_divide

def get_divide_from_bit_divide(bit_divide):
    fields_num = int(bit_divide.sum() + 1)
    resource_num = int(bit_divide.shape[0] - bit_divide.sum())
    return_divide = np.zeros((fields_num))
    resource = 0
    field = 0
    for i in range(bit_divide.shape[0]):
        if bit_divide[i] == 0:
            resource += 1
        else:
            return_divide[field] = resource
            resource = 0
            field += 1
    return_divide[field] = resource
    return return_divide

def find_last_one_index(divide):
    index = 0
    for i in range(divide.shape[0]):
        if(divide[i] == 1):
            index = i
    return index

def find_last_zero_index(divide):
    index = 0
    for i in range(divide.shape[0]):
        if(divide[i] == 0):
            index = i
    return index


def get_all_bit_divides(A, n):
    vector = np.ones((A + n - 1))
    last_bit_div = np.zeros((A + n - 1))
    vector[n - 1 :] = 0
    last_bit_div[A : ] = 1
    divides = []
    divides.append(vector)
    # print(vector)
    # print(last_bit_div)
    while(not (vector == last_bit_div).all() ):
        vector = next_bit_divide(vector)
        divides.append(vector)
    return np.array(divides)

def get_all_divides(A, n):
    bit_divides = get_all_bit_divides(A, n)
    return_divides = []
    for i in range(bit_divides.shape[0]):
        return_divides.append(get_divide_from_bit_divide(bit_divides[i]))
    return np.array(return_divides)

def payoff(divide_A, divide_B, fields_values):
    wins_A = (divide_A - divide_B > 0)
    wins_B = (divide_A - divide_B < 0)
    A_value = np.multiply(wins_A, fields_values).sum()
    B_value = np.multiply(wins_B, fields_values).sum()
    return -(int(A_value > B_value) - int(B_value > A_value))

def poly_size(number):
    return 1 #int(math.log(number) ** 2)

def get_row(row_strategy, col_strategies, fields_values): # Ta funckja będzie zwracać wiersz macierzy wypłat
    row = np.array([payoff(row_strategy, col_strategies[i], fields_values) for i in range(col_strategies.shape[0])])
    return row

def get_col(row_strategies, col_strategy, fields_values): # Ta funckja będzie zwracać wiersz macierzy wypłat
    col = np.array([payoff(col_strategy, row_strategies[i], fields_values) for i in range(row_strategies.shape[0])]).reshape(row_strategies.shape[0], 1)
    return col

def best_pure_responses_indexes_rows(columns, column_strat, number):
    # print(columns.shape)
    # print(column_strat.shape)
    row_payoffs = -np.matmul(columns, column_strat)
    indexes = np.argsort(row_payoffs, axis=0)
    return indexes[-number:]

def get_sorted_indexes(indexes):
    list_indexes = list(indexes)
    list_indexes.sort()
    return list_indexes

def add_column(columns, column, column_index):
    if(columns.shape[0] == 0):
        return column
    new_columns = np.zeros((columns.shape[0], columns.shape[1] + 1))
    new_columns[:, : column_index] = columns[:, : column_index]
    new_columns[:, column_index] = column[:,0]
    new_columns[:, column_index + 1: ] = columns[:, column_index : ]
    return new_columns

def get_rows(row_strategies, column_strategies, fields_values):
    rows = [get_row(row_strategies[i], column_strategies, fields_values) for i in range(row_strategies.shape[0])]
    rows = np.array(rows)
    return rows



def payoff_matrix(A, B, n, fields_values):
    row_strategies = get_all_divides(A, n)
    column_strategies = get_all_divides(B, n)
    rows_num = row_strategies.shape[0]
    cols_num = column_strategies.shape[0]
    matrix = np.zeros((rows_num, cols_num))
    for i in range(rows_num):
        for j in range(cols_num):
            matrix[i, j] = payoff(row_strategies[i], column_strategies[j], fields_values)
    return matrix

def payoff_matrix_pandas(A, B, n, fields_values):
    row_strategies = get_all_divides(A, n)
    column_strategies = get_all_divides(B, n)
    columns_names = []
    rows_names = []
    matrix = payoff_matrix(A, B, n, fields_values)
    for i in range(row_strategies.shape[0]):
        rows_names.append(str(row_strategies[i]))
    for i in range(column_strategies.shape[0]):
        columns_names.append(str(column_strategies[i]))
    df = pd.DataFrame(matrix, columns=columns_names, index=rows_names)
    return df

def normalize(vector):
    return vector/vector.sum()

# A, B, n = 9,10,5
# values = np.array([1,2,3,4,5])
# matrix = payoff_matrix(A, B, n, values)
# print(matrix.shape)
# print(MWU_game_algorithm(matrix, 1/64, 100000)[-1])

