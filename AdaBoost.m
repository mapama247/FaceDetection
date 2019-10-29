clear; clc;
%% Hyper-parameters
% Number of randomized Haar-features (start with a small number and increase when the algorithm is working)
nbrHaarFeatures = 100; % 100
% Number of training images, will be evenly split between faces and non-faces (Should be even)
nbrTrainImages = 2000; % 500 for development, 2000 for report 
% Number of weak classifiers
nbrWeakClassifiers = 50; % 30 or more

%% Load face and non-face data and plot a few examples
%  The data sets are shuffled each time you run the script to prevent a solution that is tailored to specific images.
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
title("Face image examples");
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
title("Non-face image examples");
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
title("Haar feature masks");
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets
% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2)); % 24x24x50
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks); % 25x50
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)]; % 1x50 / face=1 , non-face=-1

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end), nonfaces(:,:,(nbrTrainImages/2+1):end)); % 24x24x12738
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks); % 25x12738
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)]; % face=1 , non-face=-1

% Variable for the number of test-data.
nbrTestImages = length(yTest);
fprintf('Num train images: %s \n', num2str(nbrTrainImages));
fprintf('Num test images: %s \n', num2str(nbrTestImages));
fprintf('Num haar features: %s \n', num2str(nbrHaarFeatures));
 
%% AdaBoost training
weights     = (1/nbrTrainImages)*ones(nbrTrainImages,1);
alphas      = zeros(nbrWeakClassifiers,1);
thresholds  = zeros(nbrWeakClassifiers,1);
features    = zeros(nbrWeakClassifiers,1);
polarities  = zeros(nbrWeakClassifiers,1);
errors      = zeros(nbrWeakClassifiers,1);

for T = 1:nbrWeakClassifiers
    min_error = inf;
    for feature = 1:nbrHaarFeatures
        for threshold = xTrain(feature,:) + 0.01
            polarity  = 1;
            weak_classifier_output = WeakClassifier(threshold, polarity, xTrain(feature,:));
            error = WeakClassifierError(weak_classifier_output, weights, yTrain);
            if error>0.5
                polarity = -polarity;
                error = 1 - error;
            end
            if error < min_error
                min_error = error;
                t_min = threshold;
                p_min = polarity;
                f_min = feature;
                a_min = 0.5 * log((1 - min_error) / min_error);
                h_min = polarity*weak_classifier_output;
            end
        end    
    end
    
    weights = weights.*exp(-a_min * (yTrain .* h_min).'); %update weights
    weights(weights>0.5) = 0.5; %to limit the influence of outliers
    weights = weights ./ sum(weights); %normalize weights
    fprintf('WEIGHTS %i : %s \n', T, mat2str(weights)); %print new weights
    %figure(T+10)
    %plot(1:size(weights,1),weights) %display new weights in a graph

    %store best threshold/polarity/feature/alpha/error for each classifier
    thresholds(T) = t_min;
    polarities(T) = p_min;
    features(T) = f_min;
    alphas(T) = a_min;
    errors(T) = min_error;
end

%% Evaluate strong classifier with test data
cs_train    = zeros(length(thresholds), size(yTrain,2));
csa_train   = zeros(nbrWeakClassifiers,1); %accuracy
cs_test     = zeros(length(thresholds), size(yTest,2));
csa_test    = zeros(nbrWeakClassifiers,1); %accuracy
for i = 1:nbrWeakClassifiers
    %USING TRAINING DATA:
    ci_train = WeakClassifier(thresholds(i), polarities(i), xTrain(features(i),:));
    cs_train(i,:) = alphas(i) * ci_train; %calculate the alpha-weighted classification for each weak classifier
    csa_train(i) = sum(sign(sum(cs_train,1)) == yTrain) / size(sum(cs_train,1),2); %accuracy of strong classifier using i weak classifiers
    %USING TRAIN DATA:
    ci_test = WeakClassifier(thresholds(i), polarities(i), xTest(features(i),:));
    cs_test(i,:) = alphas(i) * ci_test; %calculate the alpha-weighted classification for each weak classifier
    csa_test(i) = sum(sign(sum(cs_test,1)) == yTest) / size(sum(cs_test,1),2); %accuracy of strong classifier using i weak classifiers
end
strong_classifier = sign(sum(cs_test,1)); %sum and sign the matrix over all weak classifiers.

performance = -ones(length(xTest),1); %a -1 means correct classification
performance(yTest~=strong_classifier)=1; %a 1 means incorrect classification
faces_performance    = performance(1:(size(performance,1)/2),1); %first half
nonfaces_performance = performance((size(performance,1)/2):end,1); %second half

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
figure(4)
plot(1:nbrWeakClassifiers,1-csa_test)
title('Strong Classifier')
xlabel('Num. Weak Classifiers')
ylabel('Error')

%% Plot some of the misclassified faces and non-faces from the test set
figure(5)
idxs = find(faces_performance == 1); %array with indexes of face misclassifications
suptitle('Misclassified Faces')
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,idxs(k)));
    axis image;
    axis off;
end

figure(6)
idxs = find(nonfaces_performance == 1); %array with indexes of non-face misclassifications
suptitle('Misclassified Non-Faces')
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,idxs(k)));
    axis image;
    axis off;
end

%% Plot your choosen Haar-features
figure(7)
suptitle('Choosen Haar-Features')
chosen_haars = unique(features);
ploted_haars = size(chosen_haars,1);
if ploted_haars>25
    ploted_haars = 25;
end
for k=1:ploted_haars
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,chosen_haars(k)),[-1 2]);
    axis image;
    axis off;
end

%% Display the accuracy
acc = sum(performance==-1)/size(performance,1); %should be higher than 0.9
fprintf("Accuracy: %s \n",num2str(acc));

%Plot the test accuracy and the training accuracy of the strong classifier in the same figure.
figure(8)
plot(1:nbrWeakClassifiers,csa_test,'Color','b')
hold on
plot(1:nbrWeakClassifiers,csa_train,'Color','r')
%biased but useful to check for overfitting and the validity of hyperparameters
hold off
legend('Accuracy with test data','Accuracy with training data')
title('Strong Classifier Accuracy')
xlabel('Num. Weak Classifiers')
ylabel('Accuracy')