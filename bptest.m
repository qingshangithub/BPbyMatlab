function [resultfinal, accuracy] = bptest(moxing, testData, testLabels)
    testNum = size(testData, 1);
    inWeight = moxing.inWeight;
    outWeight = moxing.outWeight;
    inBias = moxing.inBias;
    outBias = moxing.outBias;
    
    errornum = 0;
    resultfinal = zeros(testNum, 1);
    
    for i = 1:testNum
        testInput = testData(i,:);
        lab = testLabels(i,:);
        
        tmp = testInput*inWeight + inBias';
        hidResult = 1./(1+exp(-tmp));
        
        tmpp = hidResult*outWeight + outBias';
        outResult = exp(tmpp);
        outResult = outResult./sum(outResult);
        
        [~, x1] = max(outResult);
        [~, x2] = max(lab); 
        if x1 ~= x2
            errornum = errornum + 1;
        end
        resultfinal(i, 1) = x1-1;
    end
    accuracy = 1-errornum/testNum;
end