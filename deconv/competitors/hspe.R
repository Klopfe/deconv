################################################################################################################################################
## Script name: hspe.R
## Project: Deconvolution SSVR
## Description: Code function that will be called in python to perform deconvolution process with package hspe
## Author: Quentin Klopfenstein
## Date: 20/09/2021
################################################################################################################################################
library(hspe)

hspe_perso = function(Reference, Signature, Mixture, Phenotype){
  Reference = Reference[order(rownames(Reference)),]
  Mixture = Mixture[order(rownames(Mixture)),]
  Reference = Reference[rownames(Reference)%in%rownames(Mixture),]
  Mixture = Mixture[rownames(Mixture)%in%rownames(Reference),]
  pure_samples = list()
  for(i in 1:nrow(Phenotype)){
    pure_samples[[i]] = which(Phenotype[i,] == 1)
    names(pure_samples)[[i]] = rownames(Phenotype)[i]
  }
  print(pure_samples)
  out = hspe(Y=t(log2(Mixture)),references = t(log2(Reference)),
                pure_samples = pure_samples)
  return(out$estimates)
}