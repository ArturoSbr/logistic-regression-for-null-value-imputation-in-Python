import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as logit

def logit_imputer(X, y, random_state=None):
    '''
    Trains a logit model to impute missing values of an independent continuous variable. The value to replace null values with
    corresponds to the missing population's event rate.
    
        Parameters
        ----------
                X : array_like
                    Input array or object that can be converted to an array. Input the independent variable here,
                    including missing values. The length of `X` must be equal to that of `y`.
                y : array_like
                    Input array or object that can be converted to an array. Input the dichotomous dependent variable
                    here. It must not contain missing values and has to be binary (0 implies non-event and 1 implies event).
                    The length of `y` must be equal to that of `y`.
                random_state : int, default `None`
                    Integer that sets the seed prior to initiating the logistic regression.
        Returns
        -------
                num
                    Single numeric value. If the predicted value lies below or above the minimum and maximum values observed
                    in `X`, the function will return the 5th or 95th percentile accordingly, rather than returning a value
                    below or above the observed range.
        Dependencies
        ------------
                numpy, pandas, sklearn
        GitHub
        ------
                Visit https://github.com/ArturoSbr for more content.
    '''
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression as logit
    t = pd.DataFrame({'target':y,'feature':X})
    clf = logit(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, random_state=random_state,
                solver='lbfgs', max_iter=100)
    X_train = t.loc[t['feature'].notna(), 'feature'].values.reshape(-1, 1)
    y_train = t.loc[t['feature'].notna(), 'target']
    clf.fit(X_train, y_train)
    odds = max([t.loc[t['feature'].notna(), 'target'].mean(), 0.001]) / max([1 - t.loc[t['feature'].notna(), 'target'].mean(), 0.001])
    rep = (np.log(odds) - clf.intercept_[0]) / max([clf.coef_.item(0), 0.001])
    if rep < np.min(X_train):
        rep = np.percentile(X_train, 5)
    elif rep > np.max(X_train):
        rep = np.percentile(X_train, 95)
    else:
        True
    return rep
