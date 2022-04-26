################################################################################################################################################
## Script name: hspe.R
## Project: Deconvolution SSVR
## Description: Code function that will be called in python to perform deconvolution process with package hspe
## Author: Quentin Klopfenstein
## Date: 20/09/2021
################################################################################################################################################
library(hspe)
library(matrixStats)
hspe_perso = function(Reference, Signature, Mixture, Phenotype, markers=FALSE){
  Reference = Reference[order(rownames(Reference)),]
  Mixture = Mixture[order(rownames(Mixture)),]
  Reference = Reference[rownames(Reference)%in%rownames(Mixture),]
  Mixture = Mixture[rownames(Mixture)%in%rownames(Reference),]
  pure_samples = list()
  markers_list = list()
  for(i in 1:nrow(Phenotype)){
    pure_samples[[i]] = which(Phenotype[i,] == 1)
    names(pure_samples)[[i]] = rownames(Phenotype)[i]
    if(markers == TRUE){
      temp_sig = Signature[,i]
      max_row = rowMaxs(data.matrix(Signature))
      names(temp_sig) = rownames(Signature)
      temp_sig = temp_sig[temp_sig == max_row]
      idx = which(rownames(Reference)%in%names(temp_sig))
      markers_list[[i]] = idx
      names(markers_list)[[i]] = rownames(Phenotype)[i]
    }
  }
  out = hspe(Y=t(log2(Mixture)),references = t(log2(Reference)),
                pure_samples = pure_samples, markers=markers_list)

  return(out$estimates)
}