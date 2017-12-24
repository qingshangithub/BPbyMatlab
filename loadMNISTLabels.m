function labels = loadMNISTLabels(filename)  
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing  
%the labels for the MNIST images  
fp = fopen(filename, 'rb');  
assert(fp ~= -1, ['Could not open ', filename, '']);  
  
magic = fread(fp, 1, 'int32', 0, 'ieee-be');  
assert(magic == 2049, ['Bad magic number in ', filename, '']);  
  
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');  
  
label = fread(fp, inf, 'unsigned char');  
  labels=zeros(size(label,1),10);
assert(size(label,1) == numLabels, 'Mismatch in label count');  
for i=1:size(label,1)
    labels(i,label(i)+1)=1;
end
fclose(fp);   
end  