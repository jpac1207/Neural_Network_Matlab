% This function uses the maps in parameter 'preProcessingConfig' to convert each
% element on parameter 'data' to a correpondent scalar. The returned
% values are a matrix with the scalars and a array with the last position
% of 'data' as scalars too.
function [processedData, processedLabels] = preProcessing(data, preProcessingConfig)    
    dataMatrix = zeros(size(data, 1), size(data, 2));
    labels = zeros(4, size(data, 1));
    for i = 1:size(data)        
        cellArray = strsplit(data{i}, ',');        
        buying = preProcessingConfig.buyingMap(cellArray{1});
        maint = preProcessingConfig.maintMap(cellArray{2});
        doors = preProcessingConfig.doorsMap(cellArray{3});
        persons = preProcessingConfig.personsMap(cellArray{4});
        lugBoot = preProcessingConfig.lugBootMap(cellArray{5});
        safety = preProcessingConfig.safetyMap(cellArray{6});
        label = preProcessingConfig.labelMap(cellArray{7});
        dataMatrix(i, 1) = buying;
        dataMatrix(i, 2) = maint;
        dataMatrix(i, 3) = doors;
        dataMatrix(i, 4) = persons;
        dataMatrix(i, 5) = lugBoot;
        dataMatrix(i, 6) = safety;
        for j = 1:4
            if(j == label)
                labels(j, i) = 1;
            else
                labels(j, i) = 0;
            end
        end       
    end
    processedData = dataMatrix;
    processedLabels = labels;
end