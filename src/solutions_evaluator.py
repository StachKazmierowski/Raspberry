import numpy as np
import pandas as pd

def eval_strategy(payoff_matrix, row_solution, column_solution, algoritmic_strategy, game_value=0):
    row_player_strategy = algoritmic_strategy[0]
    column_player_strategy = algoritmic_strategy[1]
    column_solution = column_solution.reshape(column_solution.shape[1], 1)
    row_vector = np.matmul(row_player_strategy, payoff_matrix)
    column_vector = np.matmul(payoff_matrix, column_player_strategy)
    column_biggest_error = np.max((abs(game_value - np.multiply(row_vector.reshape(row_vector.shape[1], row_vector.shape[0]), column_solution>0)))[column_solution>0])
    row_biggest_error = np.max(abs(game_value - np.multiply(row_solution > 0, column_vector.reshape(column_vector.shape[1], column_vector.shape[0])))[row_solution > 0])
    return column_biggest_error, row_biggest_error

def possible_payoff_increase_B(strategy_A, strategy_B, payoff_mat):
    max_B_payoff = np.matmul(strategy_A, payoff_mat).max()
    curr_B_payoff = np.matmul(np.matmul(strategy_A, payoff_mat), strategy_B)[0,0]
    return max_B_payoff - curr_B_payoff

def epsilon_value(strategy_A, strategy_B, payoff_mat):
    epsilon_B = possible_payoff_increase_B(strategy_A, strategy_B, payoff_mat)
    epsilon_A = possible_payoff_increase_B(strategy_B.transpose(), strategy_A.transpose(), -payoff_mat.transpose())
    return epsilon_A, epsilon_B