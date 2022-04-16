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

% function [processedData, processedLabels] = preProcessing(data, preProcessingConfig)    
%     dataMatrix = zeros(4, size(data, 2));
%     labels = zeros(4, 4);
%     k = 1;
%     for i = 1:size(data)       
%         if i == 1 || i == 620 || i == 1617 || i == 1728
%         cellArray = strsplit(data{i}, ',');       
%         buying = preProcessingConfig.buyingMap(cellArray{1});
%         maint = preProcessingConfig.maintMap(cellArray{2});
%         doors = preProcessingConfig.doorsMap(cellArray{3});
%         persons = preProcessingConfig.personsMap(cellArray{4});
%         lugBoot = preProcessingConfig.lugBootMap(cellArray{5});
%         safety = preProcessingConfig.safetyMap(cellArray{6});
%         label = preProcessingConfig.labelMap(cellArray{7});
%         dataMatrix(k, 1) = buying;
%         dataMatrix(k, 2) = maint;
%         dataMatrix(k, 3) = doors;
%         dataMatrix(k, 4) = persons;
%         dataMatrix(k, 5) = lugBoot;
%         dataMatrix(k, 6) = safety;
%         for j = 1:4
%             if(j == label)
%                 labels(j, k) = 1;
%             else
%                 labels(j, k) = 0;
%             end
%         end   
%         k = k +1;
%         end
%     end
%     processedData = dataMatrix;
%     processedLabels = labels;
% end

