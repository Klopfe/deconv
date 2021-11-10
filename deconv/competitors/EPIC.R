################################################################################################################################################
## Script name: EPIC.R
## Project: Deconvolution SSVR
## Description: Code function that will be called in python to perform deconvolution process with package EPIC
## Author: Quentin Klopfenstein
## Date: 21/10/2020
################################################################################################################################################
library(EPIC)

EPIC_perso = function(Reference, Signature, Mixture, Phenotype){
  sigGenes = rownames(Signature)
  if(ncol(Reference) > nrow(Phenotype)){
    true_ref = matrix(0, nrow=nrow(Reference), ncol=nrow(Phenotype))
    for(i in 1:nrow(Phenotype)){
      if(length(which(Phenotype[i,]==1)) >1){
        true_ref[, i] = rowMeans(Reference[,which(Phenotype[i,]==1)])
      }else{
        true_ref[, i] = Reference[,which(Phenotype[i,]==1)]
      }
    }
    colnames(true_ref) = rownames(Phenotype)
    rownames(true_ref) = rownames(Reference)
    ref = list(refProfiles=true_ref, sigGenes=sigGenes)
  }else{
    ref = list(refProfiles=Reference, sigGenes=sigGenes)
  }
  out = EPIC(bulk = Mixture, reference = ref)
  return(out$cellFractions[,-ncol(out$cellFractions)])
}
