from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_optimize(X_train, y_train):
    # Example parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 8, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Create a RandomForestClassifier
    rf = RandomForestClassifier()

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                            cv=5, n_jobs=-1, scoring='accuracy', verbose=2)

    # Fit grid_search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and score
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_parameters, best_score
