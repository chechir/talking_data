
### Function arguments:
# obs:   numeric vector from 1 to Nlabel
# pred:  probability matrix with Nlabel columns

mlogloss = function(obs, pred){
  obs = obs + 1
  index = cbind(1:length(obs), obs)
  prob = pred[index]
  score = -mean(log(prob))
  return(score)
}