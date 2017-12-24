clc;
trainData = loadMNISTImages('train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');
testData = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');

dim=size(trainData,2);%the dimension of data
hidNum=1000;%the number of hidden layers
labNum = size(trainLabels,2);%the number of label's kind

moxing = bp(dim,hidNum, labNum, trainData, trainLabels,20,0.7); 
[~, accuracy] = bptest(moxing,testData,testLabels);
fprintf('Accuracy:%.4f\n', accuracy);

