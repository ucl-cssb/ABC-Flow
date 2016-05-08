nspecies <- 2
d <- read.csv("data.csv", header = TRUE, sep = ",")
times <- as.numeric(d[,3])
nt <- length(times)

# Fake gating
dout <- d[ which(d[,2] == "atc" & d[,3] != 24), ]
dout.s <- dout[with(dout, order(times)), ]

dout.ss <- dout.s[, c(3, 13, 15)]
write.table(dout.ss,"flow-data-gardner.txt",row.names=F,col.names=F,quote=F,sep="\t")

