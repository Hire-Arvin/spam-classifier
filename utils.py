import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


def words_in_texts(words, texts):
    """
    Args:
        words (list): Words to find.
        texts (Series): Strings to search in.
    Returns:
        A 2D NumPy array of 0s and 1s with shape (n, d).
    """
    return np.array([texts.str.contains(word, regex=False).astype(int) for word in words]).T


def extract_features(data):
    """
    Extracts engineered features from raw email data.
    Args:
        data (DataFrame): must contain 'email' and 'subject' columns.
    Returns:
        DataFrame with additional feature columns appended.
    """
    df = data.copy()
    df['has_html'] = df['email'].apply(
        lambda x: int('<html' in x.lower() or '<a ' in x.lower()
                      or '<img' in x.lower() or '<div' in x.lower())
    )
    df['perc_capitals'] = df['email'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    df['is_reply'] = df['subject'].apply(
        lambda x: int(x.strip().lower().startswith('subject: re:'))
    )
    for word in ['click', 'money', 'free', 'offer']:
        df[word] = df['email'].apply(lambda x: x.lower().split().count(word))
    df['num_exclamations'] = df['email'].apply(lambda x: x.count('!'))
    return df


def compute_CV_error(X_train, Y_train, folds=10):
    """
    K-fold cross-validation accuracy for logistic regression.
    Args:
        X_train (numpy array): Training design matrix.
        Y_train (numpy array): Binary labels.
        folds (int): Number of folds.
    Returns:
        List of k accuracy scores, one per fold.
    """
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    kf = KFold(n_splits=folds)
    accuracies = []
    for train_idx, valid_idx in kf.split(X_train):
        model.fit(X_train[train_idx], Y_train[train_idx])
        accuracies.append(np.mean(model.predict(X_train[valid_idx]) == Y_train[valid_idx]))
    return accuracies
