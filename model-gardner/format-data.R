library(reshape)

nspecies <- 2


d <- read.table("output-data.txt")
times <- as.numeric(d[1,-c(1:4)])
nt <- length(times)
d <- d[-1,]

nd <- nrow(d)/nspecies

out <- matrix(nrow=nd*nt,ncol=(nspecies+1))
cat("created out:", dim(out), "\n")


for(i in 1:nt){

   # define start and end blocks
   inds <- nd*(i-1) + 1
   inde <- inds + nd -1   
   out[inds:inde,1] <- times[i]

   for(j in 1:nspecies){
      ddt <- as.matrix( d[which(d$V4==(j-1)), 4 + i] )
      cat("dims",i,":", dim(ddt),"\n")

      cat("\tstart/end:", inds, inde, j+1, "\n")
      cat(dim(ddt), dim(out[inds:inde,j+1]), "\n")
      out[inds:inde,j+1] <- ddt
   }
}

# Fake gating
out <- out[ which(out[,2] > 0.1 & out[,3] > 0.1), ]

cat("writing out: ", nrow(out), " data\n")

write.table(out,"flow-data-gardner.txt",row.names=F,col.names=F,quote=F,sep="\t")

