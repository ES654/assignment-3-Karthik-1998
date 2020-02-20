
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    Y_hat=y_hat.values
    Y=y.values
    coun = 0
    for i in range(y.size):
        if Y_hat[i] == Y[i]:
            coun += 1
    acc = coun/y.size
    return acc

    #pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert (y_hat.size == y.size)
    Y_hat = y_hat.values
    Y = y.values
    coun = 0
    count_y_hat = 0
    for i in range(y.size):
        if Y_hat[i] == cls:
            count_y_hat += 1
        if Y_hat[i] == cls and Y[i] == cls:
            coun += 1
    if count_y_hat==0:
        precise=1
    else:
        precise = coun/count_y_hat
    return precise

    #pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    Y_hat = y_hat.values
    Y = y.values
    coun = 0
    count_y_cls = 0
    for i in range(y.size):
        if Y[i] == cls:
            count_y_cls += 1
        if Y_hat[i] == cls and Y[i] == cls:
            coun += 1
    if count_y_cls==0:
        output=1
    else:
        output = coun/count_y_cls
    return output
    #pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    output = ((sum((y-y_hat)**2))/y.size)**(1/2)
    return output
    #pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    output = (sum(abs(y-y_hat)))/y.size
    return output

    #pass
