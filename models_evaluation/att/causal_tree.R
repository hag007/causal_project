library(grf)
library(dummies)
library(MLmetrics)
func <- function(file_name){
	df=read.csv(file_name, row.names=1, header=TRUE, stringsAsFactors=FALSE)
	df=dummy.data.frame(df, drop=FALSE)
	df=df[,!(colnames(df) %in% c("x_2A","x_21A","x_24A"))]
	set.seed(42)
	
	shuffled_indices<-sample(nrow(df))
	train=df # [shuffled_indices[1:(length(shuffled_indices) %/% 2)],]
	test=df[shuffled_indices[-(1:(length(shuffled_indices) %/% 2))],]
	print(dim(train))

	# Train a causal forest.
	X <- train[,!(colnames(train) %in% c("T","Y"))]
	W <- train[,"T"]
	Y <- train[,"Y"]
	tau.forest <- causal_forest(X, Y, W)
	
	x_seq <- seq(dim(test)[1])
	
	# Estimate treatment effects for the training data using out-of-bag prediction.
	tau.hat.oob <- predict(tau.forest)
	hist(tau.hat.oob$predictions)
	 
	# Estimate treatment effects for the test sample.
	tau.hat <- predict(tau.forest, train[,!(colnames(test) %in% c("T","Y"))])
	print(MSE(Y,tau.hat))
	# Estimate the conditional average treatment effect on the treated sample (CATT).
	# Here, we don't expect much difference between the CATE and the CATT, since
	# treatment assignment was randomized.
	att <- average_treatment_effect(tau.forest, target.sample = "treated")
	print(att)
	# return(att[1])
	
}

main <- function(){
	# func("~/repos/hw4/datasets/data1.csv")
	# func("~/repos/hw4/datasets/data2.csv")
  func("/media/hag007/Data/repos/CIML/datasets/data1_p.csv")
	func("/media/hag007/Data/repos/CIML/datasets/data2_p.csv")
	
}


