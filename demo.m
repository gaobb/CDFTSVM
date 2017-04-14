%% Demo code for training and testing the CDFTSVM on an artifical dataset
clc
clear
%% load train data
<<<<<<< HEAD
load data/synthtr
=======
load synthtr
>>>>>>> 0580ee96855f76ed12515cef67ba0c506e927734
traindata=synthtr(:,1:2);
trainlabel=synthtr(:,3)*(-2)+1;

%% load test data
<<<<<<< HEAD
load data/synthte
=======
load synthte
>>>>>>> 0580ee96855f76ed12515cef67ba0c506e927734
testdata=synthte(:,1:2);
testlabel=synthte(:,3)*(-2)+1;

% Nolinear CDFTSVM
%% seting parameters
<<<<<<< HEAD
Parameter.ker = 'rbf';
Parameter.CC = 8;
Parameter.CR = 1;
Parameter.p1 = 0.2;
Parameter.v = 10;
Parameter.algorithm = 'CD';    
Parameter.showplots = true;
=======
Parameter.ker='rbf';
Parameter.CC=8;
Parameter.CR=1;
Parameter.p1=0.2;
Parameter.v=10;
Parameter.algorithm='CD';    

>>>>>>> 0580ee96855f76ed12515cef67ba0c506e927734
%% training rbf cdftsvm
[ftsvm_struct] = ftsvmtrain(traindata,trainlabel,Parameter);

%% training rbf cdftsvm
[acc]= ftsvmclass(ftsvm_struct,testdata,testlabel);

