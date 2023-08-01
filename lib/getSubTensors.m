function [Y, sub_tensor_indices] = getSubTensors(X, sub_tensor_dims, count, overlap, is_random )
%getSubTensors v1.0.0


if isempty(is_random)
   is_random = true;
end

tensor_dims = size(X);
order  = length(tensor_dims);
patch_order = length(sub_tensor_dims);

if order > length(sub_tensor_dims)
    sub_tensor_dims(length(sub_tensor_dims)+1:order) = 0;
end

Y = zeros(1,order+1);
sub_tensor_indices = [];

sub_tensors_count = count;
if count == 0 || ~is_random
    sub_tensor_dims_temp = sub_tensor_dims;
    sub_tensor_dims_temp(sub_tensor_dims_temp==0) = 1;
    mode_part_count = ceil(tensor_dims./ceil(sub_tensor_dims_temp*(1-overlap)));  
    sub_tensors_count = prod(mode_part_count);
end

if is_random 
      
    fprintf('Obtaining %d subtensors randomly \n', sub_tensors_count);
        
    for n=1:order     
        if tensor_dims(n) == sub_tensor_dims(n)
            sub_tensor_indices(:, n) = ones(sub_tensors_count,1);
        else            
            sub_tensor_indices(:, n) = randi(tensor_dims(n) - sub_tensor_dims(n),sub_tensors_count,1);
        end
    end
else    
    
    fprintf('Obtaining %.3f overlapping %d subtensors \n', overlap, sub_tensors_count);
    
    for n=1:order        
        ind = [ 1:ceil(max(sub_tensor_dims(n),1)*(1-overlap)):tensor_dims(n)-max(sub_tensor_dims(n),1) tensor_dims(n) - max(sub_tensor_dims(n),1) + 1];
        ind_count = length(ind);
        sub_tensor_indices_count = max(1,size(sub_tensor_indices,1));
        
        rep_indices = reshape(repmat(ind,sub_tensor_indices_count,1),[ind_count*sub_tensor_indices_count,1]);
        sub_tensor_indices = repmat(sub_tensor_indices,ind_count,1);
        
        sub_tensor_indices = cat(2,sub_tensor_indices, rep_indices);
    end
end

sub_tensor_dims_cell_array = num2cell(sub_tensor_dims);

for i=1:length(sub_tensor_indices(:,1))
    
    indices_origin = num2cell(sub_tensor_indices(i, :));
    indices = cellfun(@(x,y) {x:x+max(y,1)-1}, indices_origin, sub_tensor_dims_cell_array);
    
    if  i == 1 
        Y  = X(indices{:});        
    else
        Y = cat(patch_order+1,Y, X(indices{:}));
    end
    
end

