import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc


def nested_cv_roc_analysis_DF(model, param_grid, X, y, repeat_times=20, outer_cv_splits=5, inner_cv_splits=5, prob=True):
    # Initialize lists to store results
    outer_loop_auc = []
    inner_loop_auc = []
    best_params_list = []
    repeat_no = []
    outer_fold_no = []

    # Define the outer and inner cross-validation settings
    for experiment in range(1, repeat_times + 1):
        outer_cv = StratifiedKFold(n_splits=outer_cv_splits, shuffle=True, random_state=42 + experiment)

        # Outer Cross-Validation loop
        for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Inner Cross-Validation for hyperparameter tuning
            clf = GridSearchCV(model(probability=prob), param_grid, cv=inner_cv_splits, scoring='roc_auc')
            clf.fit(X_train, y_train)

            # Best model and parameters from inner CV
            best_model = clf.best_estimator_
            best_params = clf.best_params_
            best_params_list.append(best_params)

            # Inner loop AUC and ROC curve
            inner_mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            for inner_train_index, inner_test_index in StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=42 + outer_fold).split(X_train, y_train):
                X_inner_train, X_inner_test = X_train.iloc[inner_train_index], X_train.iloc[inner_test_index]
                y_inner_train, y_inner_test = y_train.iloc[inner_train_index], y_train.iloc[inner_test_index]

                best_model.fit(X_inner_train, y_inner_train)
                y_inner_pred_prob = best_model.predict_proba(X_inner_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_inner_test, y_inner_pred_prob)
                inner_mean_tpr += np.interp(mean_fpr, fpr, tpr)
                inner_mean_tpr[0] = 0.0

            inner_mean_tpr /= inner_cv_splits
            inner_mean_tpr[-1] = 1.0
            inner_mean_auc = auc(mean_fpr, inner_mean_tpr)
            inner_loop_auc.append(inner_mean_auc)

            # Start new figure for each outer fold
            plt.figure(figsize=(8, 6))
            plt.plot(mean_fpr, inner_mean_tpr, linestyle='--', label=f'Inner CV Avg (AUC = {inner_mean_auc:.2f})')
            y_proba = best_model.predict_proba(X_test)[:, 1]
            outer_auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'Outer CV (AUC = {outer_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Chance')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Experiment {experiment}, Fold {outer_fold}: Inner vs. Outer Loop ROC Curves')
            plt.legend(loc='best')
            plt.show()

            outer_loop_auc.append(outer_auc)
            repeat_no.append(experiment)
            outer_fold_no.append(outer_fold)

    # Compile results into a DataFrame
    results_df = pd.DataFrame({
        'Repeat': repeat_no,
        'Fold in Outer Loop': outer_fold_no,
        'Best Params': best_params_list,
        'Outer Loop AUC': outer_loop_auc,
        'Inner Loop AUC': inner_loop_auc
    })

    return results_df
