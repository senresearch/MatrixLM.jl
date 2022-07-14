"""
    coef(MLM::Mlm)

Extracts coefficients from Mlm object

# Arguments 

- MLM::Mlm = Mlm object

# Value

2d array of floats

"""
function coef(MLM::Mlm)
    
    return MLM.B
end


"""
    predict(MLM::Mlm, newPredictors::Predictors=MLM.data.predictors)

Calculates new predictions based on Mlm object

# Arguments 

- MLM::Mlm = Mlm object
- newPredictors::Predictors = Predictors object. Defaults to the `data.predictors` field 
  in the Mlm object used to fit the model. 

# Value

Response object

"""
function predict(MLM::Mlm, newPredictors::Predictors=MLM.data.predictors)
    
  	# Include X and Z intercepts in new data if necessary
  	if MLM.data.predictors.hasXIntercept==true && 
       newPredictors.hasXIntercept==false
    	newPredictors.X = add_intercept(newPredictors.X)
    	newPredictors.hasXIntercept = true
    	println("Adding X intercept to newPredictors.")
  	end
  	if MLM.data.predictors.hasZIntercept==true && 
       newPredictors.hasZIntercept==false
    	newPredictors.Z = add_intercept(newPredictors.Z)
    	newPredictors.hasZIntercept = true
    	println("Adding Z intercept to newPredictors.")
  	end
    
  	# Remove X and Z intercepts in new data if necessary
  	if MLM.data.predictors.hasXIntercept==false && 
       newPredictors.hasXIntercept==true
    	newPredictors.X = remove_intercept(newPredictors.X)
    	newPredictors.hasXIntercept = false
    	println("Removing X intercept from newPredictors.")
  	end
  	if MLM.data.predictors.hasZIntercept==false && 
       newPredictors.hasZIntercept==true
    	newPredictors.Z = remove_intercept(newPredictors.Z)
    	newPredictors.hasZIntercept = false
    	println("Removing Z intercept from newPredictors.")
  	end
    
    # Calculate predictions
    return Response(calc_preds(newPredictors.X, newPredictors.Z, coef(MLM))) 
end 


"""
    fitted(MLM::Mlm)

Calculate fitted values of an Mlm object

# Arguments 

- MLM::Mlm = Mlm object

# Value

Response object

"""
function fitted(MLM::Mlm)
    
    # Call the predict function with default newPredictors
    return predict(MLM)
end


"""
    resid(MLM::Mlm, newData::RawData=MLM.data)

Calculates residuals of an Mlm object

# Arguments 

- MLM::Mlm = Mlm object
- newData::RawData = RawData object. Defaults to the `data` field in the Mlm object 
  used to fit the model. 

# Value

2d array of floats

"""
function resid(MLM::Mlm, newData::RawData=MLM.data)
    
    # Include X and Z intercepts in new data if necessary
    if MLM.data.predictors.hasXIntercept==true && 
       newData.predictors.hasXIntercept==false
        newData.predictors.X = add_intercept(newData.predictors.X)
        newData.predictors.hasXIntercept = true
        newData.p = newData.p + 1
        println("Adding X intercept to newData.")
    end
    if MLM.data.predictors.hasZIntercept==true && 
       newData.predictors.hasZIntercept==false
        newData.predictors.Z = add_intercept(newData.predictors.Z)
        newData.predictors.hasZIntercept = true
        newData.q = newData.q + 1
        println("Adding Z intercept to newData.")
    end
    
    # Remove X and Z intercepts in new data if necessary
    if MLM.data.predictors.hasXIntercept==false && 
       newData.predictors.hasXIntercept==true
        newData.predictors.X = remove_intercept(newData.predictors.X)
        newData.predictors.hasXIntercept = false
        newData.p = newData.p - 1
        println("Removing X intercept from newData.")
    end
    if MLM.data.predictors.hasZIntercept==false && 
       newData.predictors.hasZIntercept==true
        newData.predictors.Z = remove_intercept(newData.predictors.Z)
        newData.predictors.hasZIntercept = false
        newData.q = newData.q - 1
        println("Removing Z intercept from newData.")
    end
    
	# Calculate residuals
    return calc_resid(get_X(newData), get_Y(newData), get_Z(newData), 
                      coef(MLM)) 
end
