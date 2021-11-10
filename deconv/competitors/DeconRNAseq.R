################################################################################################################################################
## Script name: DeconRNAseq.R
## Project: Deconvolution SSVR
## Description: Code function that will be called in python to perform deconvolution process with package DeconRNAseq
## Author: Quentin Klopfenstein
## Date: 20/09/2021
################################################################################################################################################

library(DeconRNASeq)

decon_perso = function(Signature, Mixture){
  Mixture = as.data.frame(Mixture)
  Signature = as.data.frame(Signature)
  out = DeconRNASeq(Mixture, Signature, checksig=FALSE, use.scale = TRUE, fig = FALSE)
  results = out[[1]]
  
  return(out[[1]])
} 

