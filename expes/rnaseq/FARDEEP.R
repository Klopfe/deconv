################################################################################################################################################
## Script name: FARDEEP.R
## Project: Deconvolution SSVR
## Description: Code function that will be called in python to perform deconvolution process with package FARDEEP
## Author: Quentin Klopfenstein
## Date: 20/09/2021
################################################################################################################################################
source("/Users/qklopfenstein/Documents/these/deconv/deconv/expes/rnaseq/simulated_rna/FARDEEP/sourcecode/fardeep_function.R")
source("/Users/qklopfenstein/Documents/these/deconv/deconv/expes/rnaseq/simulated_rna/FARDEEP/sourcecode/Tuning_BIC.R")


fardeep_perso = function(Mixture, Signature){
  n     = nrow(Mixture)
  p     = ncol(Signature)
  n.col = ncol(Mixture)
  beta.fardeep = matrix(0, n.col, p)
  para  = NULL
  nout = NULL
  
  for (i in 1:n.col){
    y = Mixture [, i]
    x = as.matrix(Signature)
    k = tuningBIC(x = x, y = y, n = n, p = p, intercept = TRUE)
    para    = rbind (para, k)
    reg     = fardeep(x = x, y = y, k = k, intercept = TRUE)
    coe     = reg$beta[-1]
    beta.fardeep[i, ] = coe / sum(coe)
  }
  return(beta.fardeep)
}



