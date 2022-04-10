X = [0 1 0 1; 0 0 1 1];
Y = [0 0 0 1];
H = 4;
I = 2;
O = 1;
eta = 0.1;
maxEpochs = 1000;
acceptedError = 0.000001;
activationType = 1; % 0 for sigmoid and 1 for tanh

[hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias]  = trainMLP(I, H, O, maxEpochs, eta, acceptedError, activationType, X, Y);
prediction = testMLP(hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, activationType, [1;1]);
sprintf("%f", prediction)
real = 1 & 1

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
        Y_net = k * net_o;
        E = Y - Y_net;
             
        % backward    
        df = k * ones(size(net_o));   
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
        error = sum(E.^2)/size(Y, 2);
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

function f = activation(type, value)
    if(type == 0)
        f = logsig(value);
    else
        f = tanh(value);
    end
end

function f = activationDerivative(type, value)
    if(type == 0)
        f = logsig(value) - (logsig(value).^2);
    else
        f = 1 - (tanh(value).^2);
    end
end