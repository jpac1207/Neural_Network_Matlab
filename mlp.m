H = 5; % Number of hidden layers
I = 6; % Number of input layers
O = 4; % Number of output layers
eta = 0.0001; % Learning Rate
maxEpochs = 1000; % Number of max epochs
activationType = 1; % 0 for sigmoid and 1 for tanh in the hidden layers
numberOfTrainings = 10; % number of trainings to get the error means

preProcessingConfig.buyingMap = containers.Map({'vhigh', 'high', 'med', 'low'}, {5, 4, 3, 2});
preProcessingConfig.maintMap = containers.Map({'vhigh', 'high', 'med', 'low'}, {5, 4, 3, 2});
preProcessingConfig.doorsMap = containers.Map({'2', '3', '4', '5more'}, {2, 3, 4, 5});
preProcessingConfig.personsMap = containers.Map({'2', '4', 'more'}, {2, 4, 5});
preProcessingConfig.lugBootMap = containers.Map({'small', 'med', 'big'}, {1, 2, 3});
preProcessingConfig.safetyMap = containers.Map({'low', 'med', 'high'}, {1, 2, 3});
preProcessingConfig.labelMap = containers.Map({'unacc', 'acc', 'good', 'vgood'}, {1, 2, 3, 4});

% prediction = testMLP(hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, activationType, [1;1]);
% sprintf("%f", prediction)
% real = 1 & 1
doTraining(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, eta, activationType);

function doTraining(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, eta, activationType)
    data = readData('./data/car.data');
    [X, Y] = preProcessing(data, preProcessingConfig);
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X, Y);
    finalErrors = zeros(maxEpochs, 1);  
    finalValErrors = zeros(maxEpochs, 1);
    bestError = 1;    
    
    for i = 1:numberOfTrainings
        [hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, errors, valErrors]  = trainMLP(I, H, O, maxEpochs, eta, activationType, ...
            X_train', Y_train, X_val', Y_val);  
        finalErrors = finalErrors + errors;
        finalValErrors = finalValErrors + valErrors;
        if(errors(maxEpochs) < bestError)
            bestError = errors(maxEpochs);
            save('bestWeights.mat', 'hiddenVsInputWeights', 'hiddenVsInputBias', 'outputVsHiddenWeights', 'outputVsHiddenBias');
        end        
    end
    meanFinalErrors = (finalErrors./numberOfTrainings);
    meanFinalValErrors = (finalValErrors./numberOfTrainings);
    bestError
    meanFinalErrors(maxEpochs)
    meanFinalValErrors(maxEpochs)
    plot((1:maxEpochs), meanFinalErrors, 'o');
    hold on;
    plot((1:maxEpochs), meanFinalValErrors, 'x');
    hold off;
    legend('Média Erros Treinamento', 'Média Erros Validação');
end

function data = readData(dataPath)
    data = importdata(dataPath, ',');
end

function Y = testMLP(hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, activationType, X)
    k = 1;             
    net_h = hiddenVsInputWeights * X + hiddenVsInputBias * ones(1, size(X, 2));
    Yh = activation(activationType, net_h);
    net_o = outputVsHiddenWeights * Yh + outputVsHiddenBias * ones(1, size (Yh, 2));
    Y_net = max(exp(net_o)./sum(exp(net_o)));             
end

function [hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, finalErrors, finalValErrors] = trainMLP(I, H, O, maxEpochs, eta, ...
    activationType, X_train, Y_train, X_val, Y_val)
    currentEpoch = 1;    
    errors = zeros(maxEpochs, 1);  
    validationErrors = zeros(maxEpochs, 1);  
    % Init weights    
    Whi = rand(H, I) - 0.5;
    bias_hi = rand(H, 1) - 0.5;   
    Woh = rand (O, H) - 0.5;
    bias_oh = rand(O, 1) - 0.5;    
    
    while currentEpoch <= maxEpochs       
      
        % ------- Hidden Layer -------      
        net_h = Whi * X_train + bias_hi * ones(1, size(X_train, 2));
        Yh = activation(activationType, net_h);    
        % ------- Output Layer -------
        net_o = Woh * Yh + bias_oh * ones(1, size (Yh, 2));        
        Y_net = exp(net_o)./sum(exp(net_o));                  
        E = (-1).*sum((Y_train.*log(Y_net)));                      
        %sprintf("%f", E);   

        % ------- Validation -------
        val_net_h = Whi * X_val + bias_hi * ones(1, size(X_val, 2));
        val_Yh = activation(activationType, val_net_h);    
        val_net_o = Woh * val_Yh + bias_oh * ones(1, size (val_Yh, 2));        
        val_Y_net = exp(val_net_o)./sum(exp(val_net_o));                  
        %E_val = (-1).*sum((Y_val.*log(val_Y_net)));  
        %---------------------------
        
        % backward    
        df =  (Y_train-Y_net);
        %df
        delta_bias_oh = eta * sum((E.*df)')';
        delta_Woh = eta * (E.*df)*Yh';
        Eh = (Woh')*(E.*df);
        
        df = activationDerivative(activationType, net_h);
        delta_bias_hi = (eta) * sum((Eh.*df)')';
        delta_Whi = (eta) * (Eh.*df) * X_train';
        
        %update weights  
        Whi = Whi + delta_Whi;   
        bias_hi = bias_hi + delta_bias_hi;   
        Woh = Woh + delta_Woh;
        bias_oh = bias_oh + delta_bias_oh;        
   
        %calculate error                          
        error = sum(((Y_train .* (1-Y_net))), 'all')/size(Y_train, 2);      
        validationError = sum(((Y_val .* (1-val_Y_net))), 'all')/size(Y_val, 2);  
        %sprintf("%f", error);
        errors(currentEpoch) = error;
        validationErrors(currentEpoch) = validationError;
%        if(error < acceptedError)
%            break
%        end
       currentEpoch = currentEpoch + 1;
   end     
    
    finalErrors = errors;
    finalValErrors = validationErrors;
    hiddenVsInputWeights = Whi;
    hiddenVsInputBias = bias_hi;
    outputVsHiddenWeights = Woh;
    outputVsHiddenBias = bias_oh;
end

function [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X, Y)
    numberOfRows = size(X, 1);
    trainProportion = 0.7;
    trainRows = floor(numberOfRows * trainProportion);
    valProportion = 0.2;
    valRows = floor(numberOfRows * valProportion);
    testProportion = 0.1;
    testRows = floor(numberOfRows * testProportion);

    randIndexes = randperm(numberOfRows);
    trainIndexes = randIndexes(1:trainRows);
    initOfValRows = (trainRows + 1);
    valIndexes = randIndexes(initOfValRows:(initOfValRows + valRows));
    initOfTestRows = (initOfValRows + valRows + 1);
    testIndexes = randIndexes(initOfTestRows:(initOfTestRows + testRows));

    X_train = X(trainIndexes, :);
    Y_train = Y(:, trainIndexes);
    
    X_val = X(valIndexes, :);
    Y_val = Y(:, valIndexes);
    
    X_test = X(testIndexes, :);
    Y_test = Y(:, testIndexes);
end

% This function applies the activation function on the parameter 'value'
% according with the parameter 'type'
function f = activation(type, value)
    if(type == 0)
        f = logsig(value);
    else
        f = tanh(value);
    end
end

% This function applies the derivative of activation function on the
% parameter 'value' according with the parameter 'type'
function f = activationDerivative(type, value)
    if(type == 0)
        f = logsig(value) - (logsig(value).^2);
    else
        f = 1 - (tanh(value).^2);
    end
end