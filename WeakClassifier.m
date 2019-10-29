function C = WeakClassifier(T, P, X) % Classify images using a decision stump.
% Takes a vector X of scalars obtained by applying one Haar-feature to all training images.
% Classifies the examples using a decision stump with cut-off T and parity P.
% Returns a vector C of classifications for all examples in X.
% You are not allowed to use a loop in this function (too slow)

C = ones(1,length(X));
C(P.*X < P.*T) = -1;

end