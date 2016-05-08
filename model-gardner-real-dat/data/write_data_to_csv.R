packages = c("flowCore", "flowViz", "ggplot2", "plyr", "scales", "reshape2", "RColorBrewer","ggExtra","gridExtra")

lapply(packages, require, character.only = TRUE)
sample.info = read.csv("samples.csv", header = T, stringsAsFactors = F)
sample.info[is.na(sample.info)] <- 1.000
sample.info$sample = as.character(sample.info$sample)
sample.info$inducer = factor(sample.info$inducer, levels = c("atc","iptg"))
sample.info$repeats = factor(sample.info$repeats, levels = c("1", "2", "3"))
sample.info$times = factor(sample.info$times, levels = c("0", "1", "2","3","4","5","6","7","8","9","24"))

samples = sample.info$sample
files = list.files(pattern = "*.fcs")
extractFlowData = function(flowData, samples, sample.info) {
  require("plyr")
  data = vector("list", length(samples))
  names(data) = samples
  for (i in 1:length(samples)) {
    assign(samples[i], data.frame(exprs(flowData[[i]])))
    data[[i]] = get(samples[i])
  }
  rm(list = c(samples, "i"))
  data = ldply(data, data.frame)
  print(head(data))
  data$.id = factor(data$.id)
  data = merge(sample.info, 
               data,
               by.x = "sample",
               by.y = ".id")
}

flowData = read.flowSet(files = files, transformation = F)
colnames(flowData) = c("fscA", "sscA", "GFP-A", "fl2a", "fl3a","fl4a","fscH","sscH","GFP-H","fl2H","mCherry-H","fl4H","width","Time")
sampleNames(flowData) = samples
data = extractFlowData(flowData, samples, sample.info)
write.table(data, file = "data.csv", sep = ",", qmethod = "double")
