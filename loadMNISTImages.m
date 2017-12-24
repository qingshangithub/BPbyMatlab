function images = loadMNISTImages(filename)  
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing  
%the raw MNIST images  
  
fp = fopen(filename, 'rb');  %�ö����ƴ�
assert(fp ~= -1, ['Could not open ', filename, '']);  %��������
  
magic = fread(fp, 1, 'int32', 0, 'ieee-be');  %��һ����
assert(magic == 2051, ['Bad magic number in ', filename, '']);  
  
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');  %ieeebe ��˴�
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');  
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');  
  
images = fread(fp, inf, 'unsigned char');  %���޷���8λ����
images = reshape(images, numCols, numRows, numImages);  %��ÿ��28�У�28�У�60000��ͼƬ����ά��m*n*p
images = permute(images,[2 1 3]);  %���к���
fclose(fp);  
  
% Reshape to #pixels x #examples  
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));  
images=images';
end  