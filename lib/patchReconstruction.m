function [Y] = patchReconstruction(X, sub_tensor_indices, tensor_size )
%patchReconstruction Summary of this function goes here
%   Detailed explanation goes here


tensor_dims = size(X);
order  = length(tensor_size);

sub_tensor_size = tensor_dims(1:end-1);

if order > length(sub_tensor_size)
     sub_tensor_size(length(sub_tensor_size)+1:order) = 1;
end

Y = zeros(tensor_size);
Y_count = zeros(tensor_size);

sub_tensor_dims_cell_array = num2cell(sub_tensor_size);
x_indices = repmat({':'},1,length(tensor_dims)-1);

if isa(X,'gpuArray') 
    Y = gpuArray(Y);
end

for i=1:tensor_dims(end)
    
    indices_origin = num2cell(sub_tensor_indices(i, :));
    indices = cellfun(@(x,y) {x:x+y-1}, indices_origin, sub_tensor_dims_cell_array);
   
    new_count_Ys = Y_count(indices{:}) + 1;
    Y(indices{:})  = (Y(indices{:}).*Y_count(indices{:}) + X(x_indices{:},i))./new_count_Ys; 
    
    Y_count(indices{:}) = new_count_Ys;
    
end
fprintf("Patch reconstruction completed \n");
end

