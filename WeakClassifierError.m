function E = WeakClassifierError(C, D, Y) % Calculate the error of a single decision stump.
% Takes a vector C of classifications from a weak classifier
% Takes a vector D with weights for each example
% Takes a vector Y with desired classifications.
% Calculates the weighted error of C, using the 0-1 cost function.
% You are not allowed to use a loop in this function (too slow)

I = zeros(length(D),1);
I(Y~=C)=1; % 1's indicate misclassifications
E = D.'*I;

%E = sum((D.').*(C ~= Y));
end