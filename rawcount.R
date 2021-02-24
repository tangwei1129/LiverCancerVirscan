batch="IDX"

library(tidyverse)
# Get the files names
files = list.files(pattern="*.csv")
# First apply read.csv, then rbind
#data = do.call(cbind, lapply(files, function(x) read.csv(x,row.names=1,header=T,nrows=100 )))
data = do.call(cbind, lapply(files, function(x) read.csv(x,row.names=1,header=T)))
colnames(data)=sub("*_*\\.csv","",files)



## clean data  data.cl ##
data.cl=data 
data.cl[data.cl < 2]=0
data.cl=data.cl[rowSums(data.cl) > 0 ,]
data.cl=data.cl[,colnames(data.cl)[order(colnames(data.cl))]]
dim(data.cl)

## correlation bwt replicates ##
library(corrplot)
m.rho=cor((data.cl),method="spearman")
m=cor((data.cl),method="pearson")

m.rho=m.rho[97:192,1:96]
m=m[97:192,1:96]


pdf(paste("correlation_replicates_",batch,".pdf",sep=""))
corrplot(m.rho,tl.cex=0.5,tl.col="grey30",main="rho")
corrplot(m,tl.cex=0.5,tl.col="grey30",main="cor")

rep_cor=NULL
rep_rho=NULL
for (i in 1:96){
rep_cor[i]=m[i,i]
rep_rho[i]=m.rho[i,i]
}

par(lwd=3)
hist(rep_cor,prob=T,n=50,main="correlation of replicates,cor",lwd=2,cex.lab=1.5,cex.axis=1.5,xlim=c(0,1))
lines(density(rep_cor),col="red",lwd=3,lty=2)

hist(rep_rho,prob=T,n=50,main="correlation of replicates,rho",lwd=2,cex.lab=1.5,cex.axis=1.5,xlim=c(0,1))
lines(density(rep_rho),col="red",lwd=3,lty=2)
dev.off()


# xy plot for each sample #
xy=log10(data.cl[,1:192]+1)
lim=round(quantile(as.vector(as.matrix(xy)),0.999999))


pdf(paste("replicates_plot_",batch,".pdf",sep=""),height=8,width=12)
par(mfrow=c(8,12),mar=c(2,1.5,1.5,1))
for (i in 1:96){
plot(xy[,i]~xy[,i+96],xlim=c(1,lim),ylim=c(1,lim),xlab=paste("S",i+96,sep=""),ylab=paste("S",i,sep=""),
	main=c(paste(batch,"_S",i,sep=""),paste("cor=",round(rep_cor[i],digits=2),"rho=",round(rep_rho[i],digits=2),sep=" ")),
	pch=".",cex=0.01,col="grey60",cex.axis=0.5,cex.main=0.75,cex.lab=0.5,xaxt='n',frame.plot=F)

}
dev.off()

## read counts ##
all.count=colSums(data.frame(data))




count=data.frame(all.count)
count$rep=c(rep("rep1",96),rep("rep2",96))
count$sample=as.numeric(rep(1:96,2))
library(ggplot2)
count_bar=ggplot(count, aes(sample, all.count)) +   
  geom_bar(aes(fill = rep), position = "dodge", stat="identity")+scale_x_continuous(breaks=c(1:96))


## species ##
all.sp=sapply(data,function(x){length(which(x>0))})
library(pheatmap)

sp=data.frame(all.sp)
sp$rep=c(rep("rep1",96),rep("rep2",96))
sp$sample=as.numeric(rep(1:96,2))
library(ggplot2)
sp_bar=ggplot(sp, aes(sample, all.sp)) +   
  geom_bar(aes(fill = rep), position = "dodge", stat="identity")+scale_x_continuous(breaks=c(1:96))


## average ##
all.ave=all.count/all.sp

ave=data.frame(all.ave)
ave$rep=c(rep("rep1",96),rep("rep2",96))
ave$sample=as.numeric(rep(1:96,2))
library(ggplot2)
ave_bar=ggplot(ave, aes(sample, all.ave)) +   
  geom_bar(aes(fill = rep), position = "dodge", stat="identity")+scale_x_continuous(breaks=c(1:96))


library(gridExtra)
pdf(paste("count_sp_ave_",batch,".pdf",sep=""),height=12,width=20)
grid.arrange(count_bar,sp_bar,ave_bar,ncol=1,nrow=3)
dev.off()


## write qc stat ##
qc=cbind(all.count,all.sp,all.ave,rep_cor,rep_rho,paste(batch,rep(1:96,2),sep="_"))
write.table(qc,paste("qc_stat_",batch,".txt",sep=""),quote=F,sep="\t")



