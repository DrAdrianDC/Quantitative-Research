
# This script builds and uses a Markov Chain for predicting the next state.

# Import libraries
import pandas as pd
import numpy as np


# Set the random seed for reproducibility
np.random.seed(42)

def classify_state(return_value, threshold=0.002):
    """
    Classify the daily return into Up, Down, or Stable.
    """
    if return_value > threshold:
        return "Up"
    elif return_value < -threshold:
        return "Down"
    else:
        return "Stable"

def build_transition_matrix(data):
    """
    Build the Markov Chain transition matrix from state data.
    
    Parameters:
        data (pd.Series): Series containing states.
    
    Returns:
        pd.DataFrame: Transition matrix.
    """
    states = data.unique()
    transition_matrix = pd.DataFrame(0, index=states, columns=states)
    
    for (current_state, next_state) in zip(data, data[1:]):
        transition_matrix.loc[current_state, next_state] += 1
    
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
    return transition_matrix

def predict_markov(current_state, transition_matrix):
    """
    Predict the next state using the Markov Chain.
    
    Parameters:
        current_state (str): Current state.
        transition_matrix (pd.DataFrame): Transition matrix.
    
    Returns:
        str: Predicted state.
    """
    return np.random.choice(transition_matrix.columns, p=transition_matrix.loc[current_state])

def simulate_markov_chain(transition_matrix, start_state, days=30):
    """
    Simulate a Markov chain for a given number of days.
    
    Parameters:
        transition_matrix (pd.DataFrame): Transition matrix.
        start_state (str): Starting state.
        days (int): Number of days to simulate.
    
    Returns:
        list: Simulated sequence of states.
    """
    states = [start_state]
    current_state = start_state
    
    for _ in range(days):
        next_state = predict_markov(current_state, transition_matrix)
        states.append(next_state)
        current_state = next_state
    
    return states

if __name__ == "__main__":
    from fetch_data import fetch_stock_data

    # Fetch stock data
    data = fetch_stock_data("AAPL", "2014-01-01", "2024-12-04")
    data['State'] = data['Daily Return'].apply(classify_state)
    print(data['State'].value_counts())

    # Build transition matrix
    transition_matrix = build_transition_matrix(data['State'])
    print("Transition Matrix:")
    print(transition_matrix)

    # Simulate the Markov chain for the next 30 days
    current_state = data['State'].iloc[-1]  # Use the last state's value to start the simulation
    simulated_states = simulate_markov_chain(transition_matrix, current_state, days=30)
    print(f"Simulated States: {simulated_states}")
