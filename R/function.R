# The following function is given bu xiawenjun.

#' @title slicing transformation
#' @description slicing transformation
#' @param y, response
#' @param H, slicing nums
#' @export
slicing = function(y, H){
  chunk_size = floor(length(y)/H)
  ans = matrix(0, nrow = length(y), H)
  for (i in 1:(H-1)) {
    ans[(1+chunk_size*(i-1)):(chunk_size*i),i] = y[(1+chunk_size*(i-1)):(chunk_size*i)]
  }
  ans[(1+chunk_size*(H-1)):length(y),H] = y[(1+chunk_size*(H-1)):length(y)] # 避免除不尽
  return(ans)
}


#' @title response's transformation
#' @description response's transformation
#' @param D.training, datasets consists source and target data, refer <glmtrans> package.
#' @param K, the number of source datasets.
#' @param type, response types, 'bs' means B-splines, 'slicing' means slicing.
#' @param df, number of transform directions.
#' @param degree, used in bs method.
#' @export
transform_y = function(D.training, K=1, type='bs', df=5, degree=1){
  K = length(D.training$source)
  source_Y = list()
  if(type=='bs'){
    for(k in 1:K){
      source_Y[[k]] = bs(D.training$source[[k]]$y,df=df,degree = degree)
    }
    target_Y = bs(D.training$target$y,df=df,degree = degree)
  }else if(type=='slicing'){
    for(k in 1:K){
      source_Y[[k]] = slicing(D.training$source[[k]]$y,H=df)
    }
    target_Y = slicing(D.training$target$y,H=df) 
  }
  return (list(source_Y, target_Y))
}


#' @title sufficient dimension reduction with transfer learning.
#' @description using transfer learning to get dimension reduct space.
#' @param D.training.origin, datasets consists source and target data, refer <glmtrans> package.
#' @param source_Y, transformed source response, see <transform_y> function.
#' @param target_Y, transformed target response, see <transform_y> function.
#' @export
transfer_sdr = function(D.training.origin, source_Y, target_Y){
  D.training = D.training.origin
  H = ncol(source_Y[[1]])
  p = ncol(D.training$target$x)
  K = length(D.training$source)
  
  W = matrix(0, nrow = H, ncol = p)
  
  transfer_counts = c()
  
  for (h in 1:H) {
    for (k in 1:K) {
      D.training$source[[k]]$y = source_Y[[k]][,h]
    }
    D.training$target$y = target_Y[,h]
    fit.gaussian <- glmtrans(D.training$target, D.training$source,intercept = FALSE,detection.info = FALSE)
    W[h,] = as.numeric(fit.gaussian$beta)[-1]
    transfer_counts[h] = length(fit.gaussian$transfer.source.id)
  }
  # index = order(transfer_counts,decreasing=TRUE)[1:floor(h/2)]
  # transfer_counts = transfer_counts[index]
  # W = W[index, ]
  return(list(W=W, transfer_counts = transfer_counts))
}


#' @title sufficient dimension reduction with original lasso method.
#' @description traditional method to get dimension reduct space.
#' @param D.training.origin, datasets consists source and target data, refer <glmtrans> package.
#' @param target_Y, transformed target response, see <transform_y> function.
#' @export
lasso_w = function(D.training.origin, target_Y){
  D.training = D.training.origin
  H = ncol(target_Y)
  p = ncol(D.training$target$x)
  
  W = matrix(0, nrow = H, ncol = p)
  for (h in 1:H) {
    D.training$target$y = target_Y[,h]
    fit.lasso <- cv.glmnet(x = D.training$target$x, y = D.training$target$y, intercept=FALSE)
    W[h,] = as.numeric(coef(fit.lasso,fit.lasso$lambda.min))[-1]
  }
  return(list(W=W))
}


#' @title The difference between the estimated and real dimensionality reduction space.
#' @description The difference between the calculated and real dimensionality reduction space.
#' @param W_esti, estimated reduced dimension space.
#' @param d, true order of the reduced dimension space.
#' @param true_B, true reduced dimension space.
#' @param type, method to calculate the difference, contain 'matrix-dot', 'e-norm', 'R-sqare'.
#' @param D.training, datasets consists source and target data, refer <glmtrans> package.
#' @export
compute_diff = function(W_esti, d, true_B = B, type='matrix-dot',D.training=D.training){
  # true_w = true_w/sqrt(sum(true_w^2))
  # p = ncol(W_esti)
  M = t(W_esti) %*% W_esti
  B_hat = eigen(M)$vectors[,1:d]
  B = true_B
  # B = matrix(rep(true_w,d), nrow = p, byrow = FALSE)
  if(type=='e-norm'){
    return(eigen(t(B-B_hat)%*%(B-B_hat))$values[1])
  }else if(type=='matrix-dot'){
    return (1-abs(det(t(B) %*% B_hat)))
  }else if(type=='R-sqare'){
    B_hat = matrix(B_hat, nrow = nrow(B), ncol = d)
    x_cov = cov(D.training$target$x)
    ans = c()
    for (i in 1:d) {
      curr_max = -Inf
      for (j in 1:d) {
        num = (B_hat[,i]%*%x_cov%*%t(t(B[,j])))^2
        dom = (B_hat[,i]%*%x_cov%*%t(t(B_hat[,i])))*(B[,j]%*%x_cov%*%t(t(B[,j])))
        if((num/dom)[1,1]>curr_max){curr_max=(num/dom)[1,1]}
      }
      ans[i] = curr_max
    }
    return(mean(ans))
  }
}







