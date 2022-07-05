```julia
using MatrixLM
using DataFrames
using Random
using Plots
Random.seed!(1)
```




# General Simulation

For matrix linear model, let $Y$ be a n$\times$m response matrix, the model can be expressed as: 
$$ Y = XBZ^T+E $$
Where $X_{n \times p}$ is the matrix for main predictor and $Z_{m \times q}$ denote the matrix from extra knowledge. $E_{n \times m}$ is the error term and $B_{p \times q}$ is the matrix for main and interaction effects.

For this demo, we will simulate a dataset.

First, construct a `RawData` object consisting of the response variable `Y` and row/column predictors `X` and `Z`. All three matrices must be passed in as 2-dimensional arrays. Note that the `contr` function can be used to set up treatment and/or sum contrasts for categorical variables stored in a DataFrame. By default, `contr` generates treatment contrasts for all specified categorical variables (`"treat"`). Other options include `"sum"` for sum contrasts, `"noint"` for treatment contrasts with no intercept, and `"sumnoint"` for sum contrasts with no intercept. 

## Data Generation


```julia
# Dimensions of matrices 
n = 100
m = 250

# Number of column covariates
q = 20

# Generate data with two categorical variables and 4 numerical variables.
X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n)), DataFrame(rand(n,4),:auto))

# Convert dataframe to predicton matrix
X = Matrix(contr(X_df, [:catvar1, :catvar2], ["treat", "sum"]))

p = size(X)[2]
```

The structure of X matrix looks like this.


```julia
X
```




    100×10 Matrix{Float64}:
     0.0  0.0  0.0  0.0   1.0   0.0  0.462428   0.155668  0.251098   0.589267
     1.0  0.0  0.0  0.0   0.0   1.0  0.611215   0.928079  0.299601   0.11415
     0.0  1.0  0.0  0.0   1.0   0.0  0.96203    0.928506  0.677402   0.200676
     0.0  1.0  0.0  0.0   1.0   0.0  0.338884   0.458014  0.112663   0.345292
     0.0  0.0  1.0  0.0  -1.0  -1.0  0.319402   0.876368  0.246686   0.431121
     0.0  0.0  0.0  0.0  -1.0  -1.0  0.0912716  0.798325  0.761577   0.95697
     0.0  1.0  0.0  0.0   0.0   1.0  0.89696    0.106448  0.498046   0.361177
     0.0  1.0  0.0  0.0   0.0   1.0  0.714033   0.128388  0.227456   0.637138
     0.0  1.0  0.0  0.0   0.0   1.0  0.519449   0.470248  0.30433    0.600977
     0.0  0.0  0.0  0.0   1.0   0.0  0.914266   0.616543  0.314008   0.555287
     0.0  0.0  0.0  1.0   1.0   0.0  0.037347   0.260702  0.755382   0.0335637
     0.0  0.0  0.0  1.0  -1.0  -1.0  0.35286    0.944696  0.322645   0.738069
     1.0  0.0  0.0  0.0  -1.0  -1.0  0.853496   0.819419  0.0915618  0.209414
     ⋮                          ⋮                                    
     1.0  0.0  0.0  0.0   1.0   0.0  0.788794   0.185684  0.188601   0.280887
     0.0  0.0  0.0  1.0   0.0   1.0  0.683432   0.528965  0.72794    0.991864
     0.0  0.0  0.0  1.0   0.0   1.0  0.397377   0.509531  0.266706   0.21101
     1.0  0.0  0.0  0.0   1.0   0.0  0.499295   0.813915  0.043806   0.107441
     0.0  0.0  1.0  0.0   0.0   1.0  0.778666   0.779702  0.24179    0.030524
     0.0  1.0  0.0  0.0   0.0   1.0  0.0163769  0.182888  0.430996   0.313507
     0.0  0.0  1.0  0.0   0.0   1.0  0.33769    0.62389   0.824503   0.586398
     0.0  0.0  1.0  0.0   0.0   1.0  0.643001   0.957308  0.636767   0.169251
     0.0  0.0  0.0  1.0  -1.0  -1.0  0.762811   0.787607  0.0286074  0.67255
     0.0  0.0  1.0  0.0   1.0   0.0  0.2862     0.988846  0.217446   0.724931
     0.0  0.0  1.0  0.0  -1.0  -1.0  0.469512   0.923135  0.28953    0.033319
     0.0  0.0  1.0  0.0  -1.0  -1.0  0.732537   0.350409  0.67709    0.0250041



Randomly generate some data for column covariates `Z` and response variable `Y`


```julia
Z = rand(m,q)
B = rand(-5:5,p,q)
E = randn(n,m)
Y = X*B*transpose(Z)+E
```



Finally, using all the data we have to construct a RawData object. The usage of `RawData()` can be found in here.


```julia
# Construct a RawData object
dat = RawData(Response(Y), Predictors(X, Z))
```


## Model Study

### Model construction

The matrix linear model could be build by using RawData object directly.

Least-squares estimates for matrix linear models can be obtained by running `mlm()`. An object of type `Mlm` will be returned, with variables for the coefficient estimates (`B`), the coefficient variance estimates (`varB`), and the estimated variance of the errors (`𝜎`). By default, `mlm` estimates both row and column main effects (`X` and `Z` intercepts), but this behavior can be suppressed by setting `hasXIntercept=false` and/or `hasZIntercept=false`. Column weights for `Y` and the target type for variance shrinkage can be optionally supplied to weights and targetType, respectively.


```julia
est = mlm(dat)
```




    Mlm([-0.01480609256619357 -0.13767848816234976 … 0.0036003395572182925 -0.23386389760253998; -0.23039013378797746 4.094518009040962 … 1.1808782335048782 -4.887854740646702; … ; 0.06758849829378095 5.013132457273066 … -1.1055316740898808 -3.0406601501385357; 0.1053716503212172 -4.068574352189025 … -3.10572416631665 -2.81832636203012], [0.03970587082395105 0.00951150173460745 … 0.008331428487670267 0.007772315085707244; 0.02308605048973878 0.005530240357955592 … 0.004844114352026305 0.004519030933394371; … ; 0.024001053417012263 0.005749428396130613 … 0.005036108162926166 0.004698140241603375; 0.023565545024294303 0.005645102795235486 … 0.004944726033422002 0.00461289067068521], [1.088538580562074 0.05809360901201988 … -0.019235274399189473 0.08922572398006125; 0.05809360901201988 0.9549931540382722 … -0.02249862941865301 -0.096979928167091; … ; -0.019235274399189473 -0.02249862941865301 … 0.743856243040653 0.022340244749305873; 0.08922572398006125 -0.096979928167091 … 0.022340244749305873 0.6629954181786769], RawData(Response([-1.1261057628388147 5.488321777171364 … -0.09254775167280965 -0.7950761620566996; -14.630179712135842 -2.3629328517409585 … -14.30495769189562 -10.879557257587752; … ; 1.7971569602887394 3.359201054718633 … -8.467094261805894 -5.887300223928808; -3.6393912466812535 -2.687093484440372 … -17.128600037709294 -9.976225174158724]), Predictors([1.0 0.0 … 0.25109814442172873 0.589267068138644; 1.0 1.0 … 0.29960103249698744 0.1141499280802577; … ; 1.0 0.0 … 0.28953042830432585 0.033319039396518035; 1.0 0.0 … 0.6770898997246776 0.02500408833411838], [1.0 0.6944455272959528 … 0.601302706760397 0.480612465664496; 1.0 0.6164038480415582 … 0.11380203269928679 0.12804987242261967; … ; 1.0 0.8664576963075593 … 0.8348714053116512 0.09632528333945622; 1.0 0.7309995847386772 … 0.6828968233140088 0.305122782984216], true, true), 100, 250, 11, 21), nothing, nothing, 0.0)



### Model prediction and residuals

The coefficient estimates can be accessed using `coef(est)`. Predicted values and residuals can be obtained by calling `predict()` and `resid()`. By default, both of these functions use the same data used to fit the model. However, a new `Predictors` object can be passed into `predict()` as the `newPredictors` argument and a new `RawData` object can be passed into `resid()` as the newData argument. For convenience, `fitted(est)` will return the fitted values by calling predict with the default arguments.



```julia
esti_coef = coef(est)
```




    11×21 Matrix{Float64}:
     -0.0148061  -0.137678    0.05299     …   0.0117064   0.00360034  -0.233864
     -0.23039     4.09452     0.99388        -0.97288     1.18088     -4.88785
      0.0487429  -2.94292    -5.00719         4.92993     1.01224     -3.90959
      0.0406441   0.0762493  -1.88874         5.01635    -2.00211     -0.979967
     -0.0407087   4.1666      3.06953         4.06108    -2.94325     -2.98198
      0.0044673  -1.0096      5.00068     …   3.01333    -1.0481      -1.01393
     -0.0883015   2.98511    -2.02381        -2.98537    -1.98832     -3.97098
      0.169242   -4.96929     0.0186324       2.96975    -2.90591     -4.83364
     -0.0679493  -3.8785     -1.10064         3.90863    -0.084463     2.00453
      0.0675885   5.01313     0.918518        1.11093    -1.10553     -3.04066
      0.105372   -4.06857     0.00812919  …   4.96739    -3.10572     -2.81833




```julia
preds = predict(est)
```




    Response([0.29645427576143424 4.289214874240883 … -0.5028634296324312 -1.3522885625348648; -14.138711344176942 -3.7284967895198675 … -12.720051000137637 -10.309785724272862; … ; 0.7252867743212815 4.748110008397457 … -8.399703572123716 -5.362106174596653; -4.408515238373837 -1.5702472608831584 … -17.389454541796052 -11.042140509952162])




```julia
resids = resid(est)
```




    100×250 Matrix{Float64}:
     -1.42256    1.19911    1.19457    …  -0.253511    0.410316    0.557212
     -0.491468   1.36556    0.809136      -1.12463    -1.58491    -0.569772
      2.48558    1.95313   -0.511719       1.54307    -0.28389    -2.04259
     -0.565423   0.952614  -0.45535        0.0494781   0.504206   -0.0447455
     -0.330626   0.581555   1.14086        2.44048    -0.554822   -0.528941
     -0.239987  -0.548987  -0.429152   …  -0.741959   -0.198139   -1.06484
      0.149028  -0.255974   0.427531       0.448263    0.136481    0.0537229
      1.08662   -1.75942    0.787082      -0.903802   -1.38967     0.816261
     -0.959966   0.376879  -1.56225        0.963633    0.132464   -0.101467
      1.49475   -1.31955   -1.00835        2.0863      1.45623     0.37624
     -0.061715   0.156449  -0.0366406  …   0.697305   -0.702787   -0.636806
      0.6288     0.594948   1.80702       -1.657       1.42304     2.29995
      1.91386    1.17088   -0.354871      -0.809393    0.407731    0.136582
      ⋮                                ⋱                          
      0.239983   0.11436   -0.653902       0.365401   -0.683623    2.41945
      0.423954  -0.968162  -0.741773      -1.31802     0.479357    1.81842
      0.934819   0.828713  -0.844403   …  -0.761079    0.0233866  -0.0924235
     -0.14878   -0.191073  -0.215944       0.063405    1.07207     1.22716
      0.304081  -0.337288  -0.462564      -0.516263    0.0990888   0.557854
      0.159654   2.02575   -0.621256       1.22243     1.43504    -0.872567
      1.09613    0.695913   1.26237        0.29271     0.420821    0.188869
     -0.277521  -0.388693   1.25337    …  -0.567517   -0.942597   -0.0260278
     -2.083     -0.659671   0.69622       -0.0878361   0.247423    0.3435
      0.321907   0.542107  -0.351205       1.076      -1.78139     0.724425
      1.07187   -1.38891   -0.393754      -1.09081    -0.0673907  -0.525194
      0.769124  -1.11685    0.0304466     -0.918142    0.260855    1.06592



### t-statistics and permutation test

The t-statistics for an `Mlm` object (defined as `est.B ./ sqrt.(est.varB)`) can be obtained by running `t_stat`. By default, `t_stat` does not return the corresponding t-statistics for any main effects that were estimated by `mlm`, but they will be returned if `isMainEff=true`. 



```julia
tStats = t_stat(est)
```




    10×20 Matrix{Float64}:
      55.0593    15.5157    -13.7332  …  -13.4822   16.9667    -72.7102
     -43.2127   -85.3564    -61.2276      74.6016   15.8811    -63.506
       1.04613  -30.0836    -28.961       70.9267  -29.3495    -14.8734
      56.3955    48.2328     27.4123      56.6471  -42.5652    -44.6496
     -32.2053   185.189      68.339       99.0601  -35.7229    -35.7794
      96.4889   -75.9443    101.425   …  -99.4465  -68.6703   -141.992
     -64.0735     0.278908  -40.5377      39.4619  -40.0343    -68.9459
     -51.9953   -17.1299     15.8784      54.0004   -1.20985    29.7277
      66.1145    14.0632    -28.4093      15.099   -15.5784    -44.3614
     -54.151      0.125609   38.8427      68.1343  -44.1664    -41.4959



Permutation p-values for the t-statistics can be computed by the `mlm_perms` function. `mlm_perms` calls the more general function `perm_pvals` and will run the permutations in parallel when possible. The illustrative example below only runs 5 permutations, but a different number can be specified as the second argument. By default, the function used to permute `Y` is `shuffle_rows`, which shuffles the rows for `Y`. Alternative functions for permuting `Y`, such as `shuffle_cols`, can be passed into the argument `permFun`. `mlm_perms` calls `mlm` and `t_stat` , so the user is free to specify keyword arguments for `mlm` or `t_stat`; by default, `mlm_perms` will call both functions using their default behavior. 



```julia
nPerms = 5
tStats, pVals = mlm_perms(dat, nPerms)
```




    ([55.05931221489376 15.51565078819295 … 16.96672122743137 -72.71024931701572; -43.21273927384661 -85.35637668049156 … 15.881094031008715 -63.505952381113225; … ; 66.11451934043495 14.063176844415828 … -15.578429215966365 -44.36136430997428; -54.150981121284886 0.12560858243280124 … -44.16637589174753 -41.49585354221579], [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.8 … 0.0 0.0])




```julia
heatmap(B)
```




    
![svg](../images/heatmap_B.svg)
    




```julia
heatmap(esti_coef)
```




    
![svg](../images/heatmap_esti_coef.svg)
    




```julia
heatmap(Y)
```




    
![svg](../images/heatmap_Y.svg)
    




```julia
heatmap(X)
```




    
![svg](../images/heatmap_X.svg)
    



# For ordinal variables

For this part our X matrix has only one ordinal variable(catvar1) from 1 to 5. The encoding method for ordinal variable is `SeqDiffCoding()` from package `StatsModels`


```julia
using StatsModels
```


```julia
levels = unique(X_df.catvar1)
encoding = StatsModels.ContrastsMatrix(SeqDiffCoding(), levels).matrix
encoding_intercept = inv(hcat(ones(5,1),encoding))
X2 = reduce(vcat,transpose.(map(x -> encoding_intercept[x,:], X_df.catvar1)))
```




```julia
p = size(X2)[2]
n = 100
m = 250
q = 20
```





```julia
# Number of column covariates
Z2 = rand(m,q)
B2 = rand(-5:5,p,q)
E2 = randn(n,m)
Y2 = X2*B2*transpose(Z2)+E2
```




```julia
dat2 = RawData(Response(Y2), Predictors(X2, Z2))
```



```julia
est2 = mlm(dat2)
```



```julia
heatmap(coef(est2))
```




    
![svg](../images/heatmap_esti_coef2.svg)
    




```julia
heatmap(B2)
```




    
![svg](../images/heatmap_B2.svg)
    




```julia

```
