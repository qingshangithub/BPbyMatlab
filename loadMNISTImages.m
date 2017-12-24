function images = loadMNISTImages(filename)  
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing  
%the raw MNIST images  
  
fp = fopen(filename, 'rb');  %用二进制打开
assert(fp ~= -1, ['Could not open ', filename, '']);  %发生错误
  
magic = fread(fp, 1, 'int32', 0, 'ieee-be');  %第一个数
assert(magic == 2051, ['Bad magic number in ', filename, '']);  
  
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');  %ieeebe 大端打开
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');  
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');  
  
images = fread(fp, inf, 'unsigned char');  %以无符号8位读完
images = reshape(images, numCols, numRows, numImages);  %按每张28列，28行，60000张图片；三维，m*n*p
images = permute(images,[2 1 3]);  %先行后列
fclose(fp);  
  
% Reshape to #pixels x #examples  
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));  
images=images';
end  