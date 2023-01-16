#Automated backward elimination with p values only

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0]) #returns the length of the variable
    for i in range(0, numVars): #range function generates a list of numbers that is generally used to iterate over with for loops. range(start, stop, (optional(step) basically if you want to iterate by 1, 2, 3, etc))
        regressor_OLS = sm.OLS(y, x).fit()#fitting our x and y to a linear regression creating reggresor obj
        maxVar = max(regressor_OLS.pvalues).astype(float)#we are basically returning the max value of the p values of the regressor object. #floating integers have a decimal like 3.0, etc. 
        #we should use floats when we want to be percise and saving memory is less important
        
#we've now defined max p value and now we set up a situation to use it       
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#So you are defining this function/object backward elimination that loops through
#and makes sure p value is less than sign level
#then at the bottom you set your sign level, you set your input matrix
#and you set your final output x modeled


import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
