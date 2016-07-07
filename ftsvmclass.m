function [acc,outclass,time]= ftsvmclass(ftsvm_struct,Testdata,Testlabel)
% Function:  testing ftsvm on test data
% Input:
% ftsvm_struct         - the trained  ftsvm model
% Testdata             - test data
% Testlabel            - test label
%
% Output:
% acc                    - accuracy
% outclass               - predict label
%
%  Author: Bin-Bin Gao (csgaobb@gmail.com)
% Created on 2014.10.10
% Last modified on 2015.07.16

if ( nargin>3||nargin<2) % check correct number of arguments
    help ftsvmclass
else
    
    
    [rt,ct]=size(Testdata);
    
    tic;
    if ~isempty(ftsvm_struct.scaleData)
        scaleData=ftsvm_struct.scaleData;
        for k = 1:size(Testdata, 2)
            Testdata(:,k) = scaleData.scaleFactor(k) * ...
                (Testdata(:,k) +  scaleData.shift(k));
        end
    end
    
    
    groupString=ftsvm_struct.groupString;
    vp=ftsvm_struct.vp;
    vn=ftsvm_struct.vn;
    
    X=ftsvm_struct.X;
    
    
    kfun =ftsvm_struct.KernelFunction;
    kfunargs = ftsvm_struct.KernelFunctionArgs;
    
    fprintf('Testing ...\n');
    switch ftsvm_struct.Parameter.ker
        case 'linear'
            fp=(Testdata*vp(1:(length(vp)-1))+vp(length(vp)))./norm(vp(1:(length(vp)-1)));
            fn=(Testdata*vn(1:(length(vn)-1))+vn(length(vn)))./norm(vn(1:(length(vn)-1)));
        case 'rbf'
            K = feval(kfun,Testdata,X,kfunargs{:});
            fp=(K*vp(1:(length(vp)-1))+vp(length(vp)))./norm(vp(1:(length(vp)-1)));
            fn=(K*vn(1:(length(vn)-1))+vn(length(vn)))./norm(vn(1:(length(vn)-1)));
    end
    f=fp+fn;
    
    classified=ones(rt,1);
    classified(abs(fp)>abs(fn)) = -1;
    classified(classified == -1) = 2;
    
    outclass = classified;
    unClassified = isnan(outclass);
    [~,groupString,glevels] = grp2idx(ftsvm_struct.L);
    
    outclass = glevels(outclass(~unClassified),:);
    
    if nargin==3
        correct=sum(outclass==Testlabel);
        acc=100*correct/length(Testlabel);
        fprintf('Accuracy : %3.4f (%d/%d)\n',acc,correct,length(Testlabel));
    else
        acc=[];
        fprintf('the accuracy can not be calculated, because of lack of the labels of testing data\n');
    end
    time= toc;
end
