load cereal.mat
%X = [Calories Protein Fat Sodium Fiber Carbo Sugars Shelf Potass Vitamins];
% Take a subset from a single manufacturer
mfg1 = strcmp('G',cellstr(Mfg));
X = X(mfg1,:);
size(X)

dissimilarities = pdist(zscore(X),'cityblock');
size(dissimilarities)

[Y,stress] =... 
mdscale(dissimilarities,2,'criterion','metricstress');
stress


plot(Y(:,1),Y(:,2),'o','LineWidth',2);
gname(Name(mfg1))

