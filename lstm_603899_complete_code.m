%% get the data
bar_close=importdata('bar_close.csv'); 
length(bar_close.data)
class(bar_close.data)
data = bar_close.data(1:500,:)';

% define the cutoff
numTimeStepsTrain = floor(0.9*numel(data));

% split the whole data into train and test
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

% standardize input
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

% model parameters
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% train the neural network
net = trainNetwork(XTrain,YTrain,layers,options);

% standardize the test data
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

% initialize the net
net = predictAndUpdateState(net,XTrain);

% y test used for comparison, benchmark
YTest = dataTest(2:end);

% predict with update
YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

%  unstandardize the prediction
YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-YTest).^2))  % the way to measure the prediction error

% plot the result
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)

% realized up and down
zhangdie = [];
for i=1:numel(YPred)-1
    if YTest(i)>YTest(i+1)
        zhangdie = [zhangdie; -1];
    else zhangdie = [zhangdie; 1];
    end
end

% predictions up and down
zhangdiepred = [];
for i=1:numel(YPred)-1
    if YPred(i)>YPred(i+1)
        zhangdiepred = [zhangdiepred; -1];
    else zhangdiepred = [zhangdiepred; 1];
    end
end

number_correct = sum(zhangdie.*zhangdiepred==1);
ratio_correct = number_correct/numel(zhangdie);

% pnl
price_change = [];
for j=1:numel(dataTest)-2
    price_change = [price_change;dataTest(j+2)-dataTest(j+1)];
end
earn = zhangdiepred.*price_change;
% earn = zhangdie.*price_change;
plot(cumsum(earn));