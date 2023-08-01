%Author : Ishan Wickramsingha
%Date : 2019/10/18
%Modified : 2020/09/11
classdef StatClass
        
    properties
        iteration
        residualNorm
        column
        columnIndices
        addColumn
        activeColumnsCount
        delta    
        lambda
        time        
    end
    
    methods
        function obj = StatClass( iteration,residualNorm,column,columnIndices,addColumn,activeColumnsCount,delta,lambda,time  )

            obj.iteration = iteration;
            obj.residualNorm = residualNorm;
            obj.column = column;                        
            obj.columnIndices = columnIndices;
            obj.addColumn = addColumn;
            obj.activeColumnsCount = activeColumnsCount;
            obj.delta = delta;
            obj.lambda = lambda;
            obj.time = time;
        end
       function s = saveobj(obj)
                s.iteration = obj.iteration;
                s.residualNorm = obj.residualNorm;
                s.column = obj.column;
                s.columnIndices = obj.columnIndices;
                s.addColumn = obj.addColumn;
                s.activeColumnsCount = obj.activeColumnsCount;
                s.delta = obj.delta;
                s.lambda = obj.lambda;
                s.time = obj.time;
       end
    end
    methods (Static)
      function obj = loadobj(s)
         if isstruct(s)
            obj = StatClass(s.iteration,s.residualNorm,s.column,s.columnIndices,s.addColumn,s.activeColumnsCount,s.delta,s.lambda,s.time);
         else            
            obj = s;
         end
      end
    end
end

