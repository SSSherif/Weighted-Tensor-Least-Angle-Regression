function [x] = gpuRound(x,decimals)
%gpuRound v1.0

if isa(x,'gpuArray')  
    x = round(x*10^decimals)*10^-decimals;
else
    x = round(x,decimals);
end

end

