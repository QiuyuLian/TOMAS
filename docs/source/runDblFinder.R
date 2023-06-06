library(DoubletFinder)
library(Matrix)
library(Seurat)

read_mtx <- function(mtx_folder){
  sp_matrix_read <- readMM(file.path(mtx_folder,"matrix.mtx.gz"))
  features = read.table(file.path(mtx_folder,"features.tsv.gz"))
  barcodes = read.table(file.path(mtx_folder,"barcodes.tsv.gz"))
  sp_matrix_read@Dimnames[[1]] = features[,1]
  sp_matrix_read@Dimnames[[2]] = barcodes[,1]
  return(sp_matrix_read)
}

args = commandArgs(trailingOnly=TRUE)

mtx_folder <- args[1]
meta_path <- args[2]
dbl_rate <- as.numeric(as.numeric(args[3]))

if(!dir.exists(mtx_folder) ){
  print('Please input valid mtx_folder')
  return()
}

if(! dir.exists(meta_path)){
  print('path created to save metadata.csv.')
  dir.create(meta_path)
}

# Read data ------------------------------------------------------------------------------------------------------------------
count <- read_mtx(mtx_folder)
# count <- Read10X(mtx_folder)
print('Data loaded.')

## Pre-process Seurat object (standard) --------------------------------------------------------------------------------------
seu <- CreateSeuratObject(counts=count, min.cells = 1)
seu <- NormalizeData(seu)
seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 2000)
seu <- ScaleData(seu)
seu <- RunPCA(seu)
# seu <- RunUMAP(seu, dims = 1:10)
# seu <- FindNeighbors(seu, dims = 1:10)
# seu <- FindClusters(seu, resolution = 0.1)
# DimPlot(seu, reduction = 'umap')
print('Pre-process of Seurat object is done.')

## pK Identification (no ground-truth) ---------------------------------------------------------------------------------------
sweep.res.list <- paramSweep_v3(seu, PCs = 1:10, sct = FALSE)
sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)
bcmvn <- find.pK(sweep.stats)
pK <- bcmvn$pK[which.max(bcmvn$BCmetric)]; pK <- as.numeric(levels(pK))[pK]; pK
print('pK identification is done!')

nExp_poi <- round(dbl_rate*nrow(seu@meta.data))
seu <- doubletFinder_v3(seu, PCs = 1:10, pN = 0.25, pK, nExp = nExp_poi)

write.csv(seu@meta.data, file.path(meta_path,'DoubletFinder_out.csv'))
print('All done!')

