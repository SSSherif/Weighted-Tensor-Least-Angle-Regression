%Author : Ishan Wickramsingha
%Date : 2019/10/18
classdef Parameters
        
    properties
        iterations
        residualNorm
        lambda
        activeColumnsCount
        time
    end
    
    methods
        function obj = Parameters(iterations,residualNorm,lambda,activeColumnsCount,time)
            obj.iterations = iterations;
            obj.residualNorm = residualNorm;
            obj.lambda = lambda;                        
            obj.activeColumnsCount = activeColumnsCount;
            obj.time = time;
        end
       function s = saveobj(obj)
                s.iterations = obj.iterations;
                s.residualNorm = obj.residualNorm;
                s.lambda= obj.lambda;
                s.activeColumnsCount = obj.activeColumnsCount;
                s.time = obj.time;
       end
    end
    methods (Static)
      function obj = loadobj(s)
         if isstruct(s)
            obj = Parameters(s.iterations,s.residualNorm,s.lambda,s.activeColumnsCount,s.time);
         else            
            obj = s;
         end
      end
    end
end

