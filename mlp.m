H = 6; % Number of hidden layers
I = 6; % Number of input layers
O = 4; % Number of output layers
eta = 0.00001; % Learning Rate
maxEpochs = 5000; % Number of max epochs
acceptedError = 0.001; % Max accepted error
activationType = 0; % 0 for sigmoid and 1 for tanh in the hidden layers

preProcessingConfig.buyingMap = containers.Map({'vhigh', 'high', 'med', 'low'}, {5, 4, 3, 2});
preProcessingConfig.maintMap = containers.Map({'vhigh', 'high', 'med', 'low'}, {5, 4, 3, 2});
preProcessingConfig.doorsMap = containers.Map({'2', '3', '4', '5more'}, {2, 3, 4, 5});
preProcessingConfig.personsMap = containers.Map({'2', '4', 'more'}, {2, 4, 5});
preProcessingConfig.lugBootMap = containers.Map({'small', 'med', 'big'}, {1, 2, 3});
preProcessingConfig.safetyMap = containers.Map({'low', 'med', 'high'}, {1, 2, 3});
preProcessingConfig.labelMap = containers.Map({'unacc', 'acc', 'good', 'vgood'}, {1, 2, 3, 4});

data = readData('./data/car.data');
[X, Y] = preProcessing(data, preProcessingConfig);
%X

[hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias]  = trainMLP(I, H, O, maxEpochs, eta, acceptedError, activationType, X', Y);
% prediction = testMLP(hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, activationType, [1;1]);
% sprintf("%f", prediction)
% real = 1 & 1

function data = readData(dataPath)
    data = importdata(dataPath, ',');
end

function Y = testMLP(hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, activationType, X)
    k = 1;             
    net_h = hiddenVsInputWeights * X + hiddenVsInputBias * ones(1, size(X, 2));
    Yh = activation(activationType, net_h);
    net_o = outputVsHiddenWeights * Yh + outputVsHiddenBias * ones(1, size (Yh, 2));
    Y = k * net_o;    
end

function [hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias] = trainMLP(I, H, O, maxEpochs, eta, acceptedError, activationType, X, Y)
    currentEpoch = 1;
    k = 1;   
    errors = zeros(maxEpochs, 1);    
    % Init weights    
    Whi = rand(H, I) - 0.5;
    bias_hi = rand(H, 1) - 0.5;
    %bias_hi = rand(H, 1) * 0;
    Woh = rand (O, H) - 0.5;
    bias_oh = rand(O, 1) - 0.5;
    %bias_oh = rand(O, 1) * 0;
    
    while currentEpoch <= maxEpochs       
      
        % Hidden Layer        
        net_h = Whi * X + bias_hi * ones(1, size(X, 2));
        Yh = activation(activationType, net_h);    
        % Output Layer
        net_o = Woh * Yh + bias_oh * ones(1, size (Yh, 2));        
        Y_net = exp(net_o)./sum(exp(net_o));            
        %errorMatrix = (Y .* (1-Y_net))
        %E = (Y .* (1-Y_net))
        %errorMatrix = ((-1)/size(Y_net, 2))*(sum(Y - Y_net,  'all'));
        E = (-1).*sum((Y.*log(Y_net)));    

        %E = (-1.*(sum((Y.*log(Y_net)), 'all')))./size(errorMatrix, 2)               
        sprintf("%f", E);          
        %sum((Y.*log(Y_net)))
        %df = (-1).*(Y.*log(Y_net))
        %E = (-1).*sum((Y.*log(Y_net)), "all")              
        %df = (-1).*sum((Y.*log(Y_net)))
        Y;
        df =  (Y-Y_net);
        
        % backward    
        %df = errorMatrix;   
        %df
        delta_bias_oh = eta * sum((E.*df)')';
        delta_Woh = eta * (E.*df)*Yh';
        Eh = (Woh')*(E.*df);
        
        df = activationDerivative(activationType, net_h);
        delta_bias_hi = (eta) * sum((Eh.*df)')';
        delta_Whi = (eta) * (Eh.*df) * X';
        
        %update weights  
        Whi = Whi + delta_Whi;   
        bias_hi = bias_hi + delta_bias_hi;   
        Woh = Woh + delta_Woh;
        bias_oh = bias_oh + delta_bias_oh;        
   
        %calculate error                          
        error = sum(((Y .* (1-Y_net))), 'all')/size(Y, 2);        
        %sprintf("%f", error);
        errors(currentEpoch) = error;
       if(error < acceptedError)
           break
       end
       currentEpoch = currentEpoch + 1;
   end 
    
    plot(errors)
    hiddenVsInputWeights = Whi;
    hiddenVsInputBias = bias_hi;
    outputVsHiddenWeights = Woh;
    outputVsHiddenBias = bias_oh;
end

function f = softmaxError(Y, Y_net)
    errorMatrix = zeros(size(Y, 1), size(Y, 2));
    for i = 1:size(Y,2)        
        pos = find(Y(:, i) == 1);
        errorMatrix(pos, i) = 1 - Y_net(pos, i);        
    end
    f = errorMatrix;
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