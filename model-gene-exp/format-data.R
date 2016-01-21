library(reshape)
d <- read.table("output-data.txt")
d <- d[,-c(1:4)]

# the first row has the time points of the data
names(d) <- d[1,]

d <- d[-1,]
dd <- melt(d)

write.table(dd,"flow-data-gene-exp.txt",row.names=F,col.names=F,quote=F)