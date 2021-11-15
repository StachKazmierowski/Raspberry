import numpy as np
from solutions_evaluator import epsilon_value
np.set_printoptions(suppress=True)
SIGNIFICANCE_CONST = 10**(-50)

def MWU_game_algorithm(payoff_mat, phi=1/2, steps_number=10000):
    rows_number = payoff_mat.shape[0]
    cols_number = payoff_mat.shape[1]
    p_0 = np.ones((1, rows_number))
    p_0 = p_0/rows_number
    p_t = p_0
    j_sumed = np.zeros((cols_number, 1))
    smallest_column_payoff = 1
    p_best = p_0
    p_t_sum = np.zeros((1, rows_number))
    for i in range (steps_number):
        payoffs = np.matmul(p_t, payoff_mat)
        j_best_response = np.argmax(payoffs)
        if(payoffs[0, j_best_response] < smallest_column_payoff):
            smallest_column_payoff = payoffs[0, j_best_response]
            p_best = p_t
        j_sumed[j_best_response] += 1
        m_t = payoff_mat[:,j_best_response]
        m_t_negative = (m_t < 0)
        p_t_significant = (p_t > SIGNIFICANCE_CONST)
        to_update = np.logical_or(m_t_negative, p_t_significant[0])
        m_t_updating = np.where(to_update,m_t,0)
        p_t_updating = np.where(to_update,p_t,0)
        p_t = np.multiply((1 - phi * m_t_updating), p_t_updating)
        p_t = p_t/p_t.sum()
        p_t_sum = p_t_sum + p_t
    j_distribution = j_sumed/j_sumed.sum()
    game_value = np.matmul(np.matmul(p_best, payoff_mat), j_distribution)[0][0]
    return p_best, j_distribution, -game_value, game_value, max(epsilon_value(p_best, j_distribution, payoff_mat))