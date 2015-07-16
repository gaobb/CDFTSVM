%% A CDFTSVM demo  on artifical datases
clc
clear
% load train data
load synthtr

traindata=synthtr(:,1:2);
trainlabel=synthtr(:,3)*(-2)+1;

% load test data
load synthte

testdata=synthte(:,1:2);
testlabel=synthte(:,3)*(-2)+1;
%% Nolinear CDFTSVM
% seting parameters
Parameter.ker='rbf';
Parameter.CC=8;
Parameter.CR=1;
Parameter.p1=0.2;
Parameter.v=10;
Parameter.algorithm='CD';    

% training rbf cdftsvm
[ftsvm_struct] = ftsvmtrain(traindata,trainlabel,Parameter);
% training rbf cdftsvm
[acc]= ftsvmclass(ftsvm_struct,testdata,testlabel);
% seting parameters
 ftsvmplot(ftsvm_struct,traindata,trainlabel);
