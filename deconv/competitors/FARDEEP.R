################################################################################################################################################
## Script name: FARDEEP.R
## Project: Deconvolution SSVR
## Description: Code function that will be called in python to perform deconvolution process with package FARDEEP
## Author: Quentin Klopfenstein
## Date: 20/09/2021
################################################################################################################################################
library(FARDEEP)

fardeep_perso = function(Signature, Mixture){
  results = fardeep(Signature, Mixture, alpha1=1, alpha2=1)
  beta.fardeep = results$relative.beta
  return(beta.fardeep)
}
