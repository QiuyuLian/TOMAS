
library(Matrix)
library(DoubletCollection)

read_mtx <- function(mtx_folder){
  sp_matrix_read <- t(readMM(file.path(mtx_folder,"matrix.mtx.gz")))  #######  Notice: if adata2mtx has already transposed , here no need to perform transpose t() #######
  features = read.table(file.path(mtx_folder,"features.tsv.gz"))
  barcodes = read.table(file.path(mtx_folder,"barcodes.tsv.gz"))
  sp_matrix_read@Dimnames[[1]] = features[,1]
  sp_matrix_read@Dimnames[[2]] = barcodes[,1]
  return(sp_matrix_read)
}


count <- read_mtx('./mc_count')

methods <- c('doubletCells','cxds','bcds','hybrid','scDblFinder',
             'Scrublet','DoubletFinder') 

# calculate doublet scores
score.list <- FindScores(count, methods)

save(score.list,file="dbl_scores.RData")



##### test if set doublet rate as 0.25
doublet.list.r25 <- FindDoublets(score.list, rate=0.25) 

n_droplet <- dim(count)[2]
df_label <- data.frame(matrix(0,n_droplet,length(methods)))
colnames(df_label) <- methods
rownames(df_label) <- count@Dimnames[[2]]
for(m in methods){
  
  df_label[doublet.list.r25[[m]], m] <- 1
  
}

write.csv(df_label,file='./dbl_pred.csv',row.names = FALSE)

# Choose one dbl detection results from all the computational doublet detection methods, and save it as input for TOMAS 
# Here we use "DoubletFinder"

m <- 'DoubletFinder'
write.table(count@Dimnames[[2]][doublet.list.r25[[m]]], file='./dbl_pred_DoubletFinder.txt',sep = '\t',row.names = FALSE,col.names = FALSE,quote = FALSE)




##### test if set doublet rate as 0.1
doublet.list.r10 <- FindDoublets(score.list, rate=0.10) 

n_droplet <- dim(count)[2]
df_label <- data.frame(matrix(0,n_droplet,length(methods)))
colnames(df_label) <- methods
rownames(df_label) <- count@Dimnames[[2]]
for(m in methods){
  
  df_label[doublet.list.r10[[m]], m] <- 1
  
}

write.csv(df_label,file='./dbl_pred_r10.csv',row.names = FALSE)

# Choose one dbl detection results from all the computational doublet detection methods, and save it as input for TOMAS 
# Here we use "DoubletFinder"

m <- 'DoubletFinder'
write.table(count@Dimnames[[2]][doublet.list.r10[[m]]], file='./dbl_pred_DoubletFinder_r10.txt',sep = '\t',row.names = FALSE,col.names = FALSE,quote = FALSE)





##### test if set doublet rate as 0.079
doublet.list.r7_9 <- FindDoublets(score.list, rate=0.079) 

n_droplet <- dim(count)[2]
df_label <- data.frame(matrix(0,n_droplet,length(methods)))
colnames(df_label) <- methods
rownames(df_label) <- count@Dimnames[[2]]
for(m in methods){
  
  df_label[doublet.list.r7_9[[m]], m] <- 1
  
}

write.csv(df_label,file='./dbl_pred_r7.csv',row.names = FALSE)

# Choose one dbl detection results from all the computational doublet detection methods, and save it as input for TOMAS 
# Here we use "DoubletFinder"

m <- 'DoubletFinder'
write.table(count@Dimnames[[2]][doublet.list.r7_9[[m]]], file='./dbl_pred_DoubletFinder_r7_9.txt',sep = '\t',row.names = FALSE,col.names = FALSE,quote = FALSE)



