
## Modelling ordinal data 

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

## Case study: metabolomics analysis

# Metabolomic signatures of NAFLD

*Reference:* Study ID ST001710

## Background 

Nonalcoholic fatty liver disease (NAFLD) is a progressive liver disease that is strongly associated with type 2 diabetes.  In this demo, we will apply matrix linear models to this study.

## Libraries


```julia
using LinearAlgebra,StatsModels
using MatrixLM
using CSV, DataFrames
using StatsBase
using Random
using Plots, FreqTables
```

## Data description

The data was collected from [workbench](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&DataMode=TreatmentData&StudyID=ST001710&StudyType=MS&ResultType=1#DataTabs). After we performed [data wrangling](https://github.com/GregFa/LIVERstudyST001710/blob/main/notebooks/preprocessing/WranglingST001710.ipynb) to extract the necessary information: clinical information, metabolites profiles and metabilites attributes, three data files would be involved in this Demonstration.

For the metabolomic data(`Y`), we have totally 68 different triglycerides for totally 473 subjects. 

For each triglyceride(`Z`), we have the number of carbon atom number and total double bound number.

For each subject(`X`), we have following 7 clinical variables. 

**Variables for each subject:**    

-`T2DM` => type 2 diabetes mellitus (DummyCoding) 

-`Kleiner_Steatosis` => Stage of Non-Alcoholic Fatty Liver Disease (NAFLD)  (EffectsCoding)

-`Inflammation` => Inflammation status. (From 1 to 3)

-`NAS` => NAS score (NAFLD Activity Score) represents the sum of scores for steatosis, lobular -inflammation, and ballooning, and ranges from 0-8.   

-`Platelets_E10_9_per_L` => platelets count (10⁹/L)   

-`Liver_ALT` => alanine aminotransferase test (ALT) level   

-`Liver_AST` => aspartate aminotransferase test (AST) level   

-`AST_ALT_Ratio` => AST/ALT ratio  useful in medical diagnosis   


## Input dataset 


```julia
# loading the data
clinicalDF = CSV.read("../../data/processed/ST001710_ClinicalCovariates.csv", DataFrame)
metabolitesTG = CSV.read("../../data/processed/MetaboTG.csv", DataFrame)
refTG = CSV.read("../../data/processed/refTriglycerides.csv", DataFrame);
```


```julia
print(describe(clinicalDF))
```


```julia
print(describe(refTG)) # Total carbon number and total double bound
```

```julia
freqtable(refTG.Total_DB)
```




    10-element Named Vector{Int64}
    Dim1  │ 
    ──────┼───
    0     │  9
    1     │  9
    2     │ 13
    3     │ 11
    4     │  9
    5     │  8
    6     │  6
    7     │  2
    8     │  2
    9     │  2




```julia
freqtable(refTG.Total_C)
```




    12-element Named Vector{Int64}
    Dim1  │ 
    ──────┼───
    45    │  1
    47    │  3
    48    │  6
    49    │  4
    50    │ 10
    51    │  5
    52    │ 12
    53    │  4
    54    │ 12
    55    │  1
    56    │ 10
    58    │  3



From the frequency table above, the distribution of Z is inbalanced and we need to transform them.


```julia
# Transformation
refTG2 = copy(refTG)
refTG2.Total_DB[findall(refTG.Total_DB.>=6)] .=6;
refTG2.Total_C[findall(refTG.Total_C.<=50)] .=1;
refTG2.Total_C[findall(refTG.Total_C.>50 .&& refTG.Total_C.<=55)] .=2;
refTG2.Total_C[findall(refTG.Total_C.>55)] .=3;
```


```julia
freqtable(refTG2.Total_DB)
```




    7-element Named Vector{Int64}
    Dim1  │ 
    ──────┼───
    0     │  9
    1     │  9
    2     │ 13
    3     │ 11
    4     │  9
    5     │  8
    6     │ 12




```julia
freqtable(refTG2.Total_C)
```




    3-element Named Vector{Int64}
    Dim1  │ 
    ──────┼───
    1     │ 24
    2     │ 34
    3     │ 13



## Model Decision

Our first model would be simple, with only one variable(`T2DM`) are included into the design matrix.

### Model: T2DM


```julia
# Generate X matrix
contrasts = Dict(:T2DM => EffectsCoding(base = "N"))
frml = @formula(0 ~  T2DM).rhs
# mf = ModelFrame(@formula(y ~ 1 + Sex).rhs, dfInd)
X_1 = modelmatrix(frml, clinicalDF, hints = contrasts);
```


```julia
# Generate Z matrix
contrasts = Dict(:Total_C => StatsModels.FullDummyCoding())
frml = @formula(0 ~ Total_C).rhs
# mf = ModelFrame(@formula(y ~ 1 + Sex).rhs, dfInd)
Z = modelmatrix(frml, refTG2, hints = contrasts);
```


```julia
# Y matrix
Y = Matrix(metabolitesTG)[:, 2:end]; # Remove first column
```

`@mlmformula` are similar with the `@formula` from the package `StatsModels`. The `mlmformula` macro takes expression like `1 + a*b` to construct design matrix.

Operators that have special interpretations in this syntax are:

- `+` concatenates variables as columns when generating a model matrix.
- `&` representes an interaction between two or more variables, which corresponds to a row-wise kronecker product of the individual terms (or element-wise product if all terms involved are continuous/scalar).
- `*` expands to all main effects and interactions: `a*b` is equivalent to `a+b+a&b`, `a*b*c` to `a+b+c+a&b+a&c+b&c+a&b&c`, etc.
- `1`, `0`, and `-1` indicate the presence (for `1`) or absence (for `0` and `-1`) of an intercept column.



```julia
dat = RawData(Response(Y), Predictors(X_1, Z, false,  false)) # Build raw data object
# Matrix linear model estimation, we already add an intercept when building design matrix
est = mlm(dat, addXIntercept=false, addZIntercept=false) 
esti_coef = MatrixLM.coef(est);
```

After the model estimation, we will use permutation test to calculate t statistics and p value.


```julia
nPerms = 5
# confusing about specifying again intercept boolean
tStats, pVals = mlm_perms(dat, nPerms, addXIntercept=false, addZIntercept=false);
```


```julia
znames = ["Total_C:1" "Total_C:2" "Total_C:3"]
plot(permutedims(tStats)[:,1], markershape = :circle, legend = false, title = "T-statistics of coefficient estimation", 
    xticks = (collect(1:length(znames)), znames))
```




    
![svg](../images/lineplot.svg)
    



From the model above, the triglycerides of all carbon numbers are significantly different between the people without diabeties and with diabeties.
