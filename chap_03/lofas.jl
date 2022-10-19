# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Julia 1.7.3
#     language: julia
#     name: julia-1.7
# ---

# FileName = "isl_ch03_linReg.jl.ipynb"
#
# The code here follows the concepts, examples, labs from Chap 3 of the ISL Book, where 
# Linear Regression is discussed.
#
# The formula for Multiple Linear Regression is 
#
#
# $  Y = \beta{0} + (\beta{1}X{1})  +  (\beta{2}X{2})   + . . . +  (\beta{p}X{p}) + \epsilon $
#
#
# The basic idea is to obtain the values of the beta coefficients that maximize Y given X. For the multi-variable case, this is done via the Moore-Penrose Pseudo Matrix. Once that is done, several parameters can be used to measure the accuracy of the values obtained.
#
#

# +
import Pkg

using DelimitedFiles
using CSV
using EzXML
using Dates
using DataFrames

# using MLDatasets 
# https://juliaml.github.io/MLDatasets.jl/stable/
# This Pkg is part of the JuliaML community. 
# It provides access to several datasets (Graphs, Text, Vision, Miscellaneous}  

using CUDA

using BenchmarkTools
using LinearAlgebra
using Statistics
	
using GLM
# using Distributions # included in the GLM Pkg


# +

#mkImgDir  is the directory where MecaKognia Julia Images reside
# const mkImgDir = string(HomeDir,"/MecaKognia/Img/")
#mkCodeDir  is the directory where MecaKognia Julia code resides
# const mkCodeDir = string(HomeDir,"/MecaKognia/")

# Directory where data resides
# const datDir = string(HomeDir,"/MecaKognia/Data/")

# Dir where code resides
const islCh3Dir = "/home/juan/code/julia/isl_book/chap_03/"

# Dir where data resides
const datDir = "/home/juan/Data/ML_Data/ISL_Data/V1/"
adData = string( datDir, "AdvertisingData.csv")



# +
function ReadCSVFile( fileName::String, addOnes="NO" )::DataFrame

	if addOnes== "NO"
		df = DataFrame(CSV.File( fileName))
		return df
	end

	if addOnes== "YES"
		df1 = DataFrame(CSV.File( fileName))
		nr = nrow(df1)
		#create a df of size (nr, 1) with ones in first column
		df2 = DataFrame(ones(nr, 1), :auto)
		# return a df with ones and the content from the CSV file
		df = hcat(df2, df1)
		#rename firt column as "x0" to be consistent woth regression
		n = names(df)
		rename!(df, n[1] => "x0")
		return df
	end

end

function DfConfig( df::DataFrame, cv::Vector) :: DataFrame
	return select( df, cv)
end


# -

# Estimate the single regression coefficients for the advertising data set in ISL Book Chapter 3 to reproduce the values 
# in table 3.1 (pp.68), table 3.2 (pp.69) and tables 3.3a and 3.3b (pp. 72)
#
#          sales = β0  +  β1 × TV  +   ε
#          
#          sales = β0  +  β2 × radio +  ε
#          
#          sales = β0  +  β3 × newspaper  +  ε  

# +
df    = ReadCSVFile("/home/juan/Data/ML_Data/ISL_Data/V1/AdvertisingData.csv", "NO")

df1   = DfConfig( df, [1] )                     # get Ones
dft   = DfConfig( df, [2] )                     # get TV data
dfr   = DfConfig( df, [3] )                     # get Radio data
dfn   = DfConfig( df, [4] )                     # get Newspaper data
dfs   = DfConfig( df, [5] )                     # get Sales data
dfa   = hcat( df1, dft, dfr, dfn, dfs)          # set a DF with Y and all the columns in X 

println("The columns names in the data set are ", names(dfa))

stv = lm(@formula(Sales ~ TV ), dfa) # stv is the OLS from the GLM for Sales Given TV
println("\n\n     Sales Given TV (Table 3.2 from pp. 69 \n\n")
println( stv)

## F-statistic = (SSM/DFM) / (SSE/DFE)
n = nrow(dfa)
p = 3             # Cols = {ones, TV, Sales}
p = p - 1         # the actual number of params minus the col for X0
dfm = p-1         #  DFM is the Corrected Degrees of Freedom for the Model
dfe = n-p         # DFE is the Degrees of Freedom for the Error 

ev  = stv.model.rr.y .- stv.model.rr.mu   # this is a vector with err(i) = y(i) - y_cap(i)
ev2 = ev .^ 2                             # this is a vector with the values of ev squared
sse = sum( ev2 )                          # sse = sum(i..n)[ y(i) - y_cap(i)] ^ 2

y_ave = sum( stv.model.rr.y) / n 
ssm = stv.model.rr.mu .- y_ave           # y_cap(i) - y_ave
ssm = ssm .^ 2 
ssm = sum(ssm)                          # ssm = sum(i..n)[ y_cap(i) - y_ave] ^ 2 

tss =  stv.model.rr.y .- y_ave
tss = tss .^2
tss = sum(tss)                           # tss = sum(i..n) [ y(i) - y_ave ] ^ 2

### calculate R^2
r2 = 1.0 - (sse / tss)
println("\n\nR2  = ", r2)

###calculate RSE

rse = (sse * ( 1/(n-2) )) ^ 0.5
println("RSE = ", rse)

### calculate F-Statistic
f = (ssm/dfm)/ (sse/dfe)
println("F   = ", f)

### This formula for F-Statistic ( eq. 3.23 from the ISLR2 Book is wrong !!! )
#fn = (tss - sse) / p 
#fd = sse/(n - p - 1)
#f2 = fn / fd 
#println("Wrong F = ", f2)


# Reproduce Table 3.3a in pp. 72

str = lm(@formula(Sales ~ Radio ), dfa) # str is the OLS from the GLM for Sales Given Radio
println("\n\n     Sales Given Radio (Table 3.3a) \n\n")
println(str)

# Reproduce Table 3.3b in pp. 72
stn = lm(@formula(Sales ~ Newspaper ), dfa) # str is the OLS from the GLM for Sales Given Newspaper
println("\n\n     Sales Given Newspaper (Table 3.3b) \n\n")
println(stn)

 
# -

# Estimate the regression coefficients for the advertising data set
#
#          sales = β0  +  β1 × TV  +  β2 × radio  +  β3 × newspaper   +   ε  

# +
#=
  The code in this cell reproduces  Table 3.4, pp 74, from the ISLR_V2 book.
    The equation relating the parameters

              y = b0 + b1*TV + b2*Radio + b3*Newspaper

    The coefficients found by the code are the same as in Table 3.4.
    
    The function calls the function GLM.lm from the GLM module.
    
      ols = lm(@formula(Sales ~ TV + Radio + Newspaper), df) 

    ols is a StatsModels.TableRegressionModel obj contain the following members:
    
    ols.model => Object that contains members
    ols.mf    => Object that contains the "Model Frame" which looks like a specialized data frame
    ols.mm    +> ModelMatrix{T}

    ols.model.rr returns a GLM.Resp object. Its members are:
    ols.model.rr.mu: mean response vector or fitted value, or y_cap(i)
    ols.model.rr.offset:  
    ols.model.rr.wts: optional vector of prior frequency weights for observations
    ols.model.rr.y: The original values in the response vector, or y(i) (dependent variable)
=#

df = ReadCSVFile("/home/juan/Data/ML_Data/ISL_Data/V1/AdvertisingData.csv", "NO")
ols = lm(@formula(Sales ~ TV + Radio + Newspaper), df)
println(ols)

### F-statistic = (SSM/DFM) / (SSE/DFE)
n = nrow(df)
p = ncol(df)
p = p - 1       # the actual number of params minus the col for X0
dfm = p-1       # DFM is the Corrected Degrees of Freedom for the Model
dfe = n-p       # DFE is the Degrees of Freedom for the Error 

ev  = ols.model.rr.y .- ols.model.rr.mu   # this is a vector with err(i) = y(i) - y_cap(i)
ev2 = ev .^ 2                             # this is a vector with the values of ev squared
sse = sum( ev2 )                          # sse = sum(i..n)[ y(i) - y_cap(i)] ^ 2

y_ave = sum( ols.model.rr.y) / n 
ssm = ols.model.rr.mu .- y_ave           # y_cap(i) - y_ave
ssm = ssm .^ 2 
ssm = sum(ssm)                          # ssm = sum(i..n)[ y_cap(i) - y_ave] ^ 2 

tss =  ols.model.rr.y .- y_ave
tss = tss .^2
tss = sum(tss)                           # tss = sum(i..n) [ y(i) - y_ave ] ^ 2

### calculate R^2
r2 = 1.0 - (sse / tss)
println("R2  = ", r2)

###calculate RSE

rse = (sse * ( 1/(n-2) )) ^ 0.5
println("RSE = ", rse)

### calculate F-Statistic
f = (ssm/dfm)/ (sse/dfe)
println("F   = ", f)
println("")

### This formula for F-Statistic ( eq. 3.23 from the ISLR2 Book is wrong !!! )
fn = (tss - sse) / p 
fd = sse/(n - p - 1)
f2 = fn / fd 
println("Wrong F = ", f2)

# +
#= 

The code in this cell reproduces Table 3.6, pp 76, from the ISLR_V2 book.
The Lin Reg equation is
 
          y = b0 + b1*TV + b2*Radio + b3*Newspaper
 
The code here calls function GLM.lm from the GLM module.
     
      ols = lm(@formula(Sales ~ TV + Radio + Newspaper), df) 
 
ols is an obj of type StatsModels.TableRegressionModel. 
Details are described in the comments for AdvertisingDataTable34()

F-Test for Lin reg
  
   F-statistic = (SSM/DFM) / (SSE/DFE)
  
   SSM = sum(i..n)[ y_cap(i) - y_ave    ]^2
   SSE = sum(i..n)[ y(i)     - y_cap(i) ]^2    = RSS
   TSS = sum(i..n)[ y(i)     - y_ave    ] ^ 2
 
   n = number of nRows
   p = number of params (not counting X0)
   DFM = p-1
   DFE = n-p
 
   y_cap(i) can be obtained from ols.model.rr.mu
   y(i)     can be otained from      ols.model.rr.y
   y_ave    is calculated from y(i)

  To calculate R^2:  R^2 = 1.0 - (SSE / TSS)
  To calculate RSE:  ( RSS / (n-2) )^2
 
  NOTE: The code here reproduces the same values as those of table 3.6. 
  NOTE: Eq. 3.23 for F-Test in the ISL book is wrong ! ! !   
 
  Correct Formula:  http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm

 =#

  df    = ReadCSVFile("/home/juan/Data/ML_Data/ISL_Data/V1/AdvertisingData.csv", "NO")

  df1   = DfConfig( df, [1] )     # get Ones
  dft   = DfConfig( df, [2] )     # get TV data
  dfr   = DfConfig( df, [3] )     # get Radio data
  dfn   = DfConfig( df, [4] )     # get Newspaper data
  dfs   = DfConfig( df, [5] )     # get Sales data
  dfy   = hcat( df1, dft, dfr, dfn, dfs) 
  ols   = lm(@formula(Sales ~ TV + Radio + Newspaper), dfy)
  # println(ols)

  ### F-statistic = (SSM/DFM) / (SSE/DFE)
  n = nrow(dfy)
  p = ncol(dfy)
  p = p - 1       # the actual number of params minus the col for X0
  dfm = p-1       # DFM is the Corrected Degrees of Freedom for the Model
  dfe = n-p       # DFE is the Degrees of Freedom for the Error 

  ev  = ols.model.rr.y .- ols.model.rr.mu   # this is a vector with err(i) = y(i) - y_cap(i)
  ev2 = ev .^ 2                             # this is a vector with the values of ev squared
  sse = sum( ev2 )                          # sse = sum(i..n)[ y(i) - y_cap(i)] ^ 2

  y_ave = sum( ols.model.rr.y) / n 
  ssm = ols.model.rr.mu .- y_ave           # y_cap(i) - y_ave
  ssm = ssm .^ 2 
  ssm = sum(ssm)                          # ssm = sum(i..n)[ y_cap(i) - y_ave] ^ 2 

  tss =  ols.model.rr.y .- y_ave
  tss = tss .^2
  tss = sum(tss)                           # tss = sum(i..n) [ y(i) - y_ave ] ^ 2

  ### calculate R^2
  r2 = 1.0 - (sse / tss)
  println("R2  = ", r2)

  ###calculate RSE

  rse = (sse * ( 1/(n-2) )) ^ 0.5
  println("RSE = ", rse)

  ### calculate F-Statistic
  f = (ssm/dfm)/ (sse/dfe)
  println("F   = ", f)
  println("")

  ### This formula for F-Statistic ( eq. 3.23 from the ISLR2 Book is wrong !!! )
  fn = (tss - sse) / p 
  f2 = f2 / (sse/(n - p - 1))
  fd = sse/(n - p - 1)
  f2 = fn / fd 
  println("Wrong F = ", f2)

# +
#=
Code in this cell reproduces Table 3.9, pp 89, from the ISLR_V2 book.
The equation is 

    y = b0 + b1*TV + b2*Radio + b3*Radio*TV

The coefficients produced here are
                  Coef.  Std. Error      t  Pr(>|t|)    Lower 95%   Upper 95%
(Intercept)  6.75022     0.247871    27.23    <1e-67  6.26138      7.23906
TV           0.0191011   0.00150415  12.70    <1e-26  0.0161347    0.0220675
Radio        0.0288603   0.00890527   3.24    0.0014  0.0112979    0.0464228
RTV          0.00108649  5.24204e-5  20.73    <1e-50  0.000983114  0.00118988

which are the same as in table 3.9  ;->

The code produces the same value as the book for R2 = 96.8
=#

df = ReadCSVFile(adData, "NO")
 
# Ones,TV,Radio,Newspaper,Sales
df1   = DfConfig( df, [1] )     # get Ones
dft   = DfConfig( df, [2] )     # get TV data
dfr   = DfConfig( df, [3] )     # get Radio data
dfs   = DfConfig( df, [5] )     # get Sales data
rtv = dft[:,1] .* dfr[:,1]      # this is a vector with the product of R * TV data
dfrt = DataFrame(RTV = rtv)     # this is a dataframe with the RTV product

# PrettyPrint(dfrt)
n = names(dfrt)
rename!(dfrt, n[1] => "RTV")

dfy = hcat( df1, dft, dfr, dfrt, dfs) 
n = names(dfy)

println("Names = " ,n)

ols = lm(@formula(Sales ~ TV + Radio + RTV), dfy)
println(ols)

### Compute F-Statistic = (SSM/DFN) / (SSE/DFE) for first model
### Given by   y1 = b0 + b1*HP  

n = nrow(dfy)
p = ncol(dfy)
p = p - 1       # the actual number of params minus the col for X0
dfn = p-1
dfe = n-p

ev  = ols.model.rr.y .- ols.model.rr.mu   # this is a vector with err(i) = y(i) - y_cap(i)
ev2 = ev .^ 2                             # this is a vector with the values of ev squared
sse = sum( ev2 )                          # sse = sum(i..n)[ y(i) - y_cap(i)] ^ 2

y_ave = sum( ols.model.rr.y) / n 
ssm = ols.model.rr.mu .- y_ave           # y_cap(i) - y_ave
ssm = ssm .^ 2 
ssm = sum(ssm)                          # ssm = sum(i..n)[ y_cap(i) - y_ave] ^ 2 

tss =  ols.model.rr.y .- y_ave
tss = tss .^2
tss = sum(tss)                           # tss = sum(i..n) [ y(i) - y_ave ] ^ 2

### calculate R^2
r2 = 1.0 - (sse / tss)
println("R2  = ", r2)

###calculate RSE
rse = (sse * ( 1/(n-2) )) ^ 0.5
println("RSE = ", rse)

### calculate F-Statistic
f = (ssm/dfn)/ (sse/dfe)
println("F   = ", f)
println("")
#println("Press Enter")
#enter = readline()


# +

#=
    The code is this cell reproduces Table 3.10, pp 92, from the ISLR_V2 book.  
    
    The name of the columns in the data file are

         "ones","mpg","cylinders","displacement","horsepower","weight","acceleration","year","origin","name"

    This is an example of using standard Linear Regression with polynomial terms.
    Here the HP parameter is used twice, as HP and HP*HP. This is done because the
    first take on LinReg, using all the params to estimate mpg. When looking at a 
    chart of mpg vs hp, it appears that a quadratic relashionship is a better fit
    than a linear fit. The code here explores if this is true,  
      
    The equation used to generate the values in table 3.10 is 

    y = b0 + b1*HP + b2*HP*HP
    
    where y is mpg and  HP is Horse Power. 

    The coefficients found by this code are the same as in table 3.10.

    According to the text, the R^2 for the quadratic fit is 0.688 while the linear fit is 
    0.606. I need to confirm these numbers by using eq. 3.17 form ISL text !!!. 
    
    I used the formulas summarized on the section  "Interpreting Results" of my 
    Machine Learning notes. Write functions for RSS, TSS, DFM, DFE, R^2 in the file
    LinearRegression.jl 

=#

   df = ReadCSVFile("/home/juan/Data/ML_Data/ISL_Data/V3/ISLR2/CSV/auto.csv", "NO")
  
   df1   = DfConfig( df, [1] )     # get Ones
   dfh   = DfConfig( df, [5] )     # get horse power
   dfm   = DfConfig( df, [2] )     # get mpg
   hp2   = dfh[:,1] .* dfh[:,1]    # this is a vector with the product of HP*HP
   dfp2  = DataFrame(HP2 = hp2)     # this is a dataframe with that product 
 
   # PrettyPrint(dfrt)
   n = names(dfp2)
   rename!(dfp2, n[1] => "HP2")
 
   dfy = hcat( df1, dfh, dfp2, dfm) 
   n = names(dfy)
 
   println("Names = " ,n)
 
   ols = lm(@formula(mpg ~ horsepower + HP2), dfy)
   println(ols)
 
 
