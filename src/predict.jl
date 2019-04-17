"""
    coef(MLM)

Extracts coefficients from Mlm object

# Arguments 

- MLM = Mlm object

# Value

2d array of floats

"""
function coef(MLM::Mlm)
    
    return MLM.B
end


"""
    predict(MLM, newPredictors)

Calculates new predictions based on Mlm object

# Arguments 

- MLM = Mlm object
- newPredictors = Predictors object. Defaults to the `data.predictors` field 
  in the Mlm object used to fit the model. 

# Value

Response object

"""
function predict(MLM::Mlm, newPredictors::Predictors=MLM.data.predictors)
    
  	# Include X and Z intercepts in new data if necessary
  	if MLM.data.predictors.isXIntercept==true && 
       newPredictors.isXIntercept==false
    	newPredictors.X = add_intercept(newPredictors.X)
    	newPredictors.isXIntercept = true
    	println("Adding X intercept to newPredictors.")
  	end
  	if MLM.data.predictors.isZIntercept==true && 
       newPredictors.isZIntercept==false
    	newPredictors.Z = add_intercept(newPredictors.Z)
    	newPredictors.isZIntercept = true
    	println("Adding Z intercept to newPredictors.")
  	end
    
  	# Remove X and Z intercepts in new data if necessary
  	if MLM.data.predictors.isXIntercept==false && 
       newPredictors.isXIntercept==true
    	newPredictors.X = remove_intercept(newPredictors.X)
    	newPredictors.isXIntercept = false
    	println("Removing X intercept from newPredictors.")
  	end
  	if MLM.data.predictors.isZIntercept==false && 
       newPredictors.isZIntercept==true
    	newPredictors.Z = remove_intercept(newPredictors.Z)
    	newPredictors.isZIntercept = false
    	println("Removing Z intercept from newPredictors.")
  	end
    
    # Calculate predictions
    return Response(calc_preds(newPredictors.X, newPredictors.Z, coef(MLM))) 
end 


"""
    fitted(MLM)

Calculate fitted values of an Mlm object

# Arguments 

- MLM = Mlm object

# Value

Response object

"""
function fitted(MLM::Mlm)
    
    # Call the predict function with default newPredictors
    return predict(MLM)
end


"""
    resid(MLM, newData)

Calculates residuals of an Mlm object

# Arguments 

- MLM = Mlm object
- newData = RawData object. Defaults to the `data` field in the Mlm object 
  used to fit the model. 

# Value

2d array of floats

"""
function resid(MLM::Mlm, newData::RawData=MLM.data)
    
    # Include X and Z intercepts in new data if necessary
    if MLM.data.predictors.isXIntercept==true && 
       newData.predictors.isXIntercept==false
        newData.predictors.X = add_intercept(newData.predictors.X)
        newData.predictors.isXIntercept = true
        newData.p = newData.p + 1
        println("Adding X intercept to newData.")
    end
    if MLM.data.predictors.isZIntercept==true && 
       newData.predictors.isZIntercept==false
        newData.predictors.Z = add_intercept(newData.predictors.Z)
        newData.predictors.isZIntercept = true
        newData.q = newData.q + 1
        println("Adding Z intercept to newData.")
    end
    
    # Remove X and Z intercepts in new data if necessary
    if MLM.data.predictors.isXIntercept==false && 
       newData.predictors.isXIntercept==true
        newData.predictors.X = remove_intercept(newData.predictors.X)
        newData.predictors.isXIntercept = false
        newData.p = newData.p - 1
        println("Removing X intercept from newData.")
    end
    if MLM.data.predictors.isZIntercept==false && 
       newData.predictors.isZIntercept==true
        newData.predictors.Z = remove_intercept(newData.predictors.Z)
        newData.predictors.isZIntercept = false
        newData.q = newData.q - 1
        println("Removing Z intercept from newData.")
    end
    
	# Calculate residuals
    return calc_resid(get_X(newData), get_Y(newData), get_Z(newData), 
                      coef(MLM)) 
end
