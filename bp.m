function [moxing] = bp(a,b,c, trainData, trainLabels,cycle,L)

    dim = a; % number of input units,784
    hidNum = b; % number of hidden units,3000
    labNum = c; % number of softmax output units,10
    lRate = 0.005;
    lamda=L;%regulization
    inWeight = 0.01*randn(dim, hidNum);
    outWeight = 0.01*randn(hidNum, labNum);
    batchsize = 100; 
    batchnum = 600; 
    inBias = zeros(hidNum, 1);
    outBias = zeros(labNum, 1);

    gtoutBias=1e-8*ones(labNum,1);
    gtihw=1e-8*ones(dim,hidNum);
    gtinBias=1e-8*ones(hidNum,1);
    gthow=1e-8*ones(hidNum,labNum);

    batch1 = reshape(trainData', dim, batchsize, batchnum);
    batch2 = reshape(trainLabels', labNum, batchsize, batchnum);

    for epoch = 1:cycle
       fprintf(1, 'epoch %d\n', epoch);
       tic;

       for batch = 1:batchnum
            batch11 = batch1(:,:,batch)'; 
            batch22 = batch2(:,:,batch)';  

            tmp = batch11*inWeight + repmat(inBias, 1, batchsize)'; 
                                      
            hidResult = 1./(1+exp(-tmp));

            tmpp = hidResult*outWeight + repmat(outBias, 1, batchsize)';    
            tmpp = tmpp - repmat(max(tmpp,[],2), 1, labNum);
            outResult = exp(tmpp);
            outResult = outResult./repmat(sum(outResult,2),1,labNum); 

            outDer = outResult - batch22; 

            outWeightgd = hidResult'*outDer+lamda*outWeight; 
            outWeightdt = outWeightgd/batchnum;

            outBias_grad = sum(outDer, 1)';
            outBias_delta = outBias_grad/batchnum;
            gtoutBias=(gtoutBias.^2+outBias_delta.^2).^0.5;
            outBias = outBias - lRate*outBias_delta./gtoutBias;

            inDer = outDer*outWeight'.*hidResult.*(1-hidResult); 
            inWeight_grad = batch11'*inDer+lamda*inWeight;
            inWeight_delta = inWeight_grad/batchnum;
            gtihw=(gtihw.^2+inWeight_delta.^2).^0.5;
            inWeight = inWeight - lRate*inWeight_delta./gtihw;

            inBias_grad = sum(inDer, 1)';
            inBias_delta = inBias_grad/batchnum;
            gtinBias=(gtinBias.^2+inBias_delta.^2).^0.5;
            inBias = inBias - lRate*inBias_delta./gtinBias;

            gthow=(gthow.^2+outWeightdt.^2).^0.5;
            outWeight = outWeight - lRate*outWeightdt./gthow;

        end
        toc;   

    end

    moxing.inWeight = inWeight;
    moxing.inBias = inBias;
    moxing.outWeight = outWeight;
    moxing.outBias = outBias;
end
