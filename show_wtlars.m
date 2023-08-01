function [  ] = show_wtlars( Ax, Y, Tensor_Norm, Norm_R, Tensor_Dimensions )
%show_wtlars v1.0
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%This function display tlars results

%% Function Call

% show_tlars( Ax, Y, Tensor_Norm, Norm_R, Dim_Array );

%% Inputs 

%Variable        Type               Description

%Ax            (Numeric Array)  = Current result in a vector form
%Y             (N-D Array)      = Normalizerd input data tensor
%Tensor_Norm   (Numeric)        = Norm of the input data tensor
%Dim_Array     (Numeric Array)  = Dimensions of the core tensor as an array

%% show_tlars 
frame = 1;
image_frame = frame;
figureNumber = 2;
draw_image = false;

Y = double(Y);


if length(Tensor_Dimensions)>= 3 && Tensor_Dimensions(3) == 3
    image_frame = [1 2 3];
end

if length(Tensor_Dimensions) > 1

    Irn = reshape(Ax,Tensor_Dimensions);
    
    if length(Tensor_Dimensions) == 2
        Ir= Irn.*Tensor_Norm;
        Iry = Y.*Tensor_Norm;
        draw_image = true;
    elseif length(Tensor_Dimensions) == 3
        Ir= Irn(:,:,image_frame).*Tensor_Norm;
        Iry = Y(:,:,image_frame).*Tensor_Norm;
        draw_image = true;
    elseif length(Tensor_Dimensions) == 4
        Ir= Irn(:,:,image_frame,frame).*Tensor_Norm;
        Iry = Y(:,:,image_frame,frame).*Tensor_Norm;
        draw_image = true;
    elseif length(Tensor_Dimensions) == 5
        Ir= Irn(:,:,image_frame,frame,frame).*Tensor_Norm;
        Iry = Y(:,:,image_frame,frame,frame).*Tensor_Norm;
        draw_image = true;
    end

end


if length(Norm_R) <= 5
   figure(figureNumber);
end

fig = gcf;

if fig.Number ~= figureNumber
    r = groot;
    for i = 1:length(r.Children)    
        if r.Children(i).Number == figureNumber
           fig = r.Children(i);
        end  
    end
    set(0,'CurrentFigure',fig);
end

if draw_image
    subplot(1,3,1);
    if length(frame) == 3
        imshow(uint8(full(Iry)));
    else    
        imshow(mat2gray(double(full(Iry))));
    end
    title('Original','Interpreter','latex');
    subplot(1,3,2);
    if length(frame) == 3
        imshow(uint8(full(Ir)));
    else    
        imshow(mat2gray(double(full(Ir))));
    end
    imshow(mat2gray(double(full(Ir))));
    title('Reconstructed','Interpreter','latex');
    subplot(1,3,3);
end

plot(Norm_R);
title('$||R||_2$ Vs. Iterations','Interpreter','latex');
caption = sprintf('Iterations \n $||R||_2$ = %d', Norm_R(end));
ylabel('$||R||_2$','Interpreter','latex');
xlabel(caption,'Interpreter','latex');
drawnow

end

