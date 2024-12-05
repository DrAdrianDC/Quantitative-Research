# This script trains and evaluates a Logistic Regression model for predicting stock states.

# Import libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def train_ml_model(X, y):
    """
    Train a Logistic Regression model with class imbalance handling and cross-validation.
    
    Parameters:
        X (np.array): Feature matrix.
        y (np.array): Labels.
    
    Returns:
        model: Trained Logistic Regression model.
    """
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model with class weight balancing
    model = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr', class_weight='balanced')
    
    # Perform cross-validation
    print("Performing Cross-Validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())
    
    # Train the model on the training set
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Test Set Accuracy:", acc)
    print("Confusion Matrix on Test Set:\n", cm)
    
    return model

if __name__ == "__main__":
    from fetch_data import fetch_stock_data
    from markov_model import classify_state

    # Fetch and preprocess data
    data = fetch_stock_data("AAPL", "2014-01-01", "2024-12-04")
    data['State'] = data['Daily Return'].apply(classify_state)
    
    # Map states to numerical labels
    state_map = {"Up": 2, "Down": 0, "Stable": 1}
    data['State_Num'] = data['State'].map(state_map)

    X = data[['Daily Return']].values[1:]  # Feature matrix
    y = data['State_Num'].values[1:]      # Labels
    
    # Train the model
    model = train_ml_model(X, y)
