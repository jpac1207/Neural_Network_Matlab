% ---------- Parâmetros Gerais ----------
maxEpochs = 1500; % Número de épocas do treinamento
activationType = 1; % Flag para escolha de função de ativação dos neurônios escondidos. 0 para sigmoid e 1 para tanh.
numberOfTrainings = 10; % Número de treinamentos a serem utilizad
H = 15; % Número de neurônios na camada escondida
I = 6; % Número de neurônios na camada de entrada
O = 4; % Número de neurônios na camada de saída
eta = 0.05; % Learning Rateos para computar as médias.

% ---------- Mapas a serem utilizados no pré processamento de dados ----------
preProcessingConfig.buyingMap = containers.Map({'vhigh', 'high', 'med', 'low'}, {5, 4, 3, 2});
preProcessingConfig.maintMap = containers.Map({'vhigh', 'high', 'med', 'low'}, {5, 4, 3, 2});
preProcessingConfig.doorsMap = containers.Map({'2', '3', '4', '5more'}, {2, 3, 4, 5});
preProcessingConfig.personsMap = containers.Map({'2', '4', 'more'}, {2, 4, 5});
preProcessingConfig.lugBootMap = containers.Map({'small', 'med', 'big'}, {1, 2, 3});
preProcessingConfig.safetyMap = containers.Map({'low', 'med', 'high'}, {1, 2, 3});
preProcessingConfig.labelMap = containers.Map({'unacc', 'acc', 'good', 'vgood'}, {1, 2, 3, 4});


%testRow = 1212;
%predictExampleUsingBestWeights(preProcessingConfig, activationType, testRow);

% ---------- Chamadas de funções para computação de métricas ----------

% Realiza treinamento da MLP 'numberOfTrainings' vezes.
doTraining(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, eta, activationType);

% Realiza treinamento da MLP 'numberOfTrainings' vezes variando o número de neurônios da camada escondida.
%doTrainingWithHiddenLayerSizeVariation(preProcessingConfig, maxEpochs, numberOfTrainings, I, 5, 15, O, eta, activationType);

% Realiza treinamento da MLP 'numberOfTrainings' vezes variando a taxa de aprendizado.
%doTrainingWithEtaVariation(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, [0.0001 0.00001 0.000001 0.0000001 0.00000001], activationType)   

% ---------- Implementações das funções de computação de métricas ----------

% Realiza 'numberOfTrainings' treinamentos, obtendo ao final:
% Melhor erro de treinamento encontrado
% Média dos erros de treinamento
% Média dos erros de validação
% Gráfico com os erros médios por epóca
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
    meanFinalError = meanFinalErrors(maxEpochs)
    meanFinalValError = meanFinalValErrors(maxEpochs)
    plot((1:maxEpochs), meanFinalErrors, 'o');
    hold on;
    plot((1:maxEpochs), meanFinalValErrors, 'x');
    hold off;
    legend('Média Erros Treinamento', 'Média Erros Validação');
end

% Realiza 'numberOfTrainings' treinamentos, variando a quantidade de neurônios da camada escondida ['H_init', 'H_end']. Obtendo ao final:
% Melhor erro de treinamento encontrado
% Média dos erros de treinamento
% Média dos erros de validação
% Gráfico com os erros médios por epóca
function doTrainingWithHiddenLayerSizeVariation(preProcessingConfig, maxEpochs, numberOfTrainings, I, H_init, H_end, O, eta, activationType)
   H = H_init;
   while H <= H_end
    H
    doTraining(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, eta, activationType);
    H = H+1;
    pause;
   end
end

% Realiza 'numberOfTrainings' treinamentos, variando a taxa de aprendizado em função dos elementos do vetor 'etas'. Obtendo ao final:
% Melhor erro de treinamento encontrado
% Média dos erros de treinamento
% Média dos erros de validação
% Gráfico com os erros médios por epóca
function doTrainingWithEtaVariation(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, etas, activationType)   
   i = 1;  
   while i <= size(etas, 2)
       etas(i)
       doTraining(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, etas(i), activationType);
       i = i+1;
       pause;
   end
end

% Realiza predição do exemplo da linha 'rowOfExample' do dataset,
% utilizando os pesos salvos no arquivo 'bestWeights.mat', que deve se
% encontrar no mesmo diretório do arquivo aqui executado
function predictExampleUsingBestWeights(preProcessingConfig, activationType, rowOfExample)
    weightsStruct = load('bestWeights.mat');
    hiddenVsInputWeights = weightsStruct.hiddenVsInputWeights;
    hiddenVsInputBias = weightsStruct.hiddenVsInputBias;
    outputVsHiddenWeights = weightsStruct.outputVsHiddenWeights;
    outputVsHiddenBias = weightsStruct.outputVsHiddenBias;
    data = readData('./data/car.data');
    [X, Y] = preProcessing(data, preProcessingConfig);    
    prediction = testMLP(hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, activationType, X(rowOfExample, :)');
    [~, real] = max(Y(:, rowOfExample));
    sprintf("Predição: %d", prediction)
    sprintf("Real: %d", real)
end

% Realiza predição da classe de um dado padrão de entrada 'X', utilizando
% os parâmetros: 
% hiddenVsInputWeights -> Matriz que representa os pesos aprendidos para as
% conexões entre 
function Y = testMLP(hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, activationType, X)          
    net_h = hiddenVsInputWeights * X + hiddenVsInputBias * ones(1, size(X, 2));
    Yh = activation(activationType, net_h);
    net_o = outputVsHiddenWeights * Yh + outputVsHiddenBias * ones(1, size (Yh, 2));
    Y_net = exp(net_o)./sum(exp(net_o));
    [value, index] = max(Y_net);
    Y = index;
end

% Realiza o treinamento da MLP, de acordo com os parametros:
% I -> Número de neurônios na camada de entrada
% H -> Número de neurônios na camada escondida
% O -> Número de neurônios na camada de saída
% maxEpochs -> Número de epócas do treinamento
% eta -> Taxa de aprendizado
% activationType -> Flag utilizada para definir a função de ativação da
% camada escondida
% X_train -> Padrões de entrada utilizados durante o treinamento
% Y_train -> Padrões de saída utilizados durante o treinamento
% X_val -> Padrões de entrada utilizados na validação
% Y_val -> Padrões de saída utilizados na validação
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
        Y_net = exp(net_o)./sum(exp(net_o));   % Aplicação da softmax              
        E = ((-1).*sum((Y_train.*log(Y_net))))./size(Y_train, 2);  % Computação do erro                   
        %sprintf("%f", E);   

        % ------- Validation -------
        val_net_h = Whi * X_val + bias_hi * ones(1, size(X_val, 2));
        val_Yh = activation(activationType, val_net_h);    
        val_net_o = Woh * val_Yh + bias_oh * ones(1, size (val_Yh, 2));        
        val_Y_net = exp(val_net_o)./sum(exp(val_net_o));                          
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
        error = sum(((Y_train .* (1-Y_net)).^2), 'all')/size(Y_train, 2);     
        validationError = sum(((Y_val .* (1-val_Y_net))), 'all')/size(Y_val, 2);  
        %sprintf("%f", error);
        errors(currentEpoch) = error;
        validationErrors(currentEpoch) = validationError;
       currentEpoch = currentEpoch + 1;
   end     
    
    finalErrors = errors;
    finalValErrors = validationErrors;
    hiddenVsInputWeights = Whi;
    hiddenVsInputBias = bias_hi;
    outputVsHiddenWeights = Woh;
    outputVsHiddenBias = bias_oh;
end

% Realiza o carregamento dos dados contidos no arquivo existente no caminho
% 'dataPath'
function data = readData(dataPath)
    data = importdata(dataPath, ',');
end

% Realiza a divisão dos dados contidos em 'X' e 'Y' em:
% X_train -> Padrões de entrada a serem utilizados no treino (70%)
% Y_train -> Padrões de saída a serem utilizados no treino (70%)
% X_val -> Padrões de entrada a serem utilizados na validação (20%)
% Y_val -> Padrões de saída a serem utilizados na validação (20%)
% X_test -> Padrões de entrada a serem utilizados no teste (10%)
% Y_test -> Padrões de saída a serem utilizados no testw (10%)
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
    valIndexes = randIndexes(initOfValRows:(initOfValRows + valRows -1));
    initOfTestRows = (initOfValRows + valRows);
    testIndexes = randIndexes(initOfTestRows:(initOfTestRows + testRows-1));

    X_train = X(trainIndexes, :);
    Y_train = Y(:, trainIndexes);
    
    X_val = X(valIndexes, :);
    Y_val = Y(:, valIndexes);
    
    X_test = X(testIndexes, :);
    Y_test = Y(:, testIndexes);
end

% Aplica uma função de ativação no parâmetro 'value' utilizando a flag 'type'
function f = activation(type, value)
    if(type == 0)
        f = logsig(value);
    else
        f = tanh(value);
    end
end

% Computa a derivada de 'value' utilizando a função definida pela flag 'type'
function f = activationDerivative(type, value)
    if(type == 0)
        f = logsig(value) - (logsig(value).^2);
    else
        f = 1 - (tanh(value).^2);
    end
end