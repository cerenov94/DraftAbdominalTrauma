import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from configs import Config

class ParticipantVisibleError(Exception):
    pass



def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df
def create_training_solution(y_train):
    sol_train = y_train.copy()

    # bowel healthy|injury sample weight = 1|2
    sol_train['bowel_weight'] = np.where(sol_train['bowel_injury'] == 1, 2, 1)

    # extravasation healthy/injury sample weight = 1|6
    sol_train['extravasation_weight'] = np.where(sol_train['extravasation_injury'] == 1, 6, 1)

    # kidney healthy|low|high sample weight = 1|2|4
    sol_train['kidney_weight'] = np.where(sol_train['kidney_low'] == 1, 2,
                                          np.where(sol_train['kidney_high'] == 1, 4, 1))

    # liver healthy|low|high sample weight = 1|2|4
    sol_train['liver_weight'] = np.where(sol_train['liver_low'] == 1, 2, np.where(sol_train['liver_high'] == 1, 4, 1))

    # spleen healthy|low|high sample weight = 1|2|4
    sol_train['spleen_weight'] = np.where(sol_train['spleen_low'] == 1, 2,
                                          np.where(sol_train['spleen_high'] == 1, 4, 1))

    #any healthy|injury sample weight = 1|6
    #sol_train['any_injury_weight'] = np.where(sol_train['any_injury'] == 1, 6, 1)
    return sol_train


def log_score(y, prediction):
    ## full correct Y's and Y prediction
    y = np.concatenate(y)
    prediction = np.concatenate(prediction)

    y = post_proc(y)
    prediction = post_proc(prediction)


    ## sample submission target columns
    true_columns = ['bowel_healthy','bowel_injury',
                  'extravasation_healthy','extravasation_injury',
                  'kidney_healthy','kidney_low','kidney_high',
                  'liver_healthy','liver_low','liver_high',
                  'spleen_healthy','spleen_low','spleen_high']

    ## creating Y and Y pred. dataframes
    submission = pd.DataFrame(prediction,columns=true_columns)
    solution = pd.DataFrame(y,columns=true_columns)
    ## adding weights
    solution = create_training_solution(solution)
    solution[true_columns] = solution[true_columns].astype(np.int64)



    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets

    label_group_losses = []
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')
        submission = normalize_probabilities_to_one(submission, col_group)

        label_group_losses.append(
            sklearn.metrics.log_loss(
                y_true=solution[col_group].values.astype(np.int64),
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        )

    # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    #healthy_cols = [x + '_healthy' for x in all_target_categories]
    # any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
    #
    # any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)

    return np.mean(label_group_losses)


def post_proc(pred):
    proc_pred = np.empty((pred.shape[0], 2 * 2 + 3 * 3), dtype="float32")

    # bowel, extravasation
    proc_pred[:, 1] = pred[:, 0]
    proc_pred[:, 0] = 1 - proc_pred[:, 1]
    proc_pred[:, 3] = pred[:, 1]
    proc_pred[:, 2] = 1 - proc_pred[:, 3]

    # liver, kidney, sneel
    proc_pred[:, 4:7] = pred[:, 2:5]
    proc_pred[:, 7:10] = pred[:, 5:8]
    proc_pred[:, 10:13] = pred[:, 8:11]
    #proc_pred[:,-1] = pred[:,-1]

    return proc_pred

def reshaping_true(y):
    #proc_pred = np.empty((pred.shape[0], 2 * 2 + 3 * 3), dtype="float32")
    y_true = np.empty((y.shape[0],14),dtype = 'float32')
    # bowel, extravasation
    y_true[:, 1] = y[:, 0]
    y_true[:, 0] = 1 - y_true[:, 1]
    y_true[:, 3] = y[:, 1]
    y_true[:, 2] = 1 - y_true[:, 3]

    # liver, kidney, sneel
    y_true[:, 4:7] = y[:, 2:5]
    y_true[:, 7:10] = y[:, 5:8]
    y_true[:, 10:13] = y[:, 8:11]
    # any injury
    y_true[:,-1] = y[:,-1]

    return y_true



# 1. y_true and y_pred -> true_df,pred_df
# 2. true_df -> create_training_solution(true_df) WEIGHTS
# 3. log_score(true_df,pred_df)