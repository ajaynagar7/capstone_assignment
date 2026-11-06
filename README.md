Express service type is delayed most often with rate 48.97% (delay_rate = 0.4897).
from tier 1 to tier 3 delay rate increases continuously from 0.04 in tier 1, 0.215 in tier 2 and 0.44 in tier 3.
more the distance more is the delay_rate for 0-200kms band it is 0.0219 for 200-400kms band it is 0.1368 for 400-600kms band it is 0.4364 and for 600+ delay rate is heighest 0.8523.
the three columns most strongly correlated with delivery_hours are promised_hours      0.8372, distance_km 0.4013, n_stops 0.3247.
we performed log transformation on declared value because it can be right scewed means may have higher tail datas so median and mead are differ highly without log transformation like before log transformation mean is 1533.0542 and median is 1121.5 while after log transformation mean is 7.0289 and median is 7.0233 which are close enough.

#"Model results (regression)".
Baseline R2 is -0.0212 which is close to zero, it represents that if we assign all the predicted values to mean of target variable than its r2 is 0.

best model on validation is linear regression ranked 1 with Lowest RMSE 4.8912hours.
depths 1 and 2 are high biased have high RMSE on training as well as validataion prediction and depts unlimited, 12, 8 are high variance as difference between training prediction RMSE and validation prediction RMSE is larger predictining good on training but failing to predictt unknown data, memorising data
learning rate 0.1 is best 0.01 is too slow to reaching to the right value of prediction and 1 is overshooting is keep misses the point and reacheds to other side of graph
I will ship linear regression in that case because it has good and comparable score with the grandientboodting and it is fast enogh so the gape between scores is so small can be ignored 

#"Model results (classification)"
Blind_base_accuracy is 0.7667 is useless because it has no logic because it is predicting all reaches at on time without any logic.
the noise columns' combined share of the forest's importance is 0.2191 yes f_score for top 7 seven very high in comparision to all five noise columns 
From K table for k = 7 val_f1 for randomforest model is 0.7848 and after tuning with k value 7 is is giving f1_score 0.8333 which is IMPROVED
final test result shows predicted 30 is delayed accuratly but missed 11 actully delayed to predict and predicted on time it's not as good prediction when FN costs more 

