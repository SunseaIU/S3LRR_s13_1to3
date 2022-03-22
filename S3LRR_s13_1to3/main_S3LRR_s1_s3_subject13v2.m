% semi-supervised sparse low-rank regression (S3LRR)

% Yong Peng @ Hangzhou Dianzi University
% October, 2021

clear; close all; clc;

% data preparation:  session 1 -> session 3
load('fea_session1_subject13.mat');
load('gnd_session1.mat');
[d,ntr]= size(fea);
c = length(unique(gnd));

XL = fea;

YL = zeros(ntr,c);
for i = 1:ntr
    YL(i,gnd(i)) = 1;
end
clear fea gnd

load('fea_session3_subject13.mat');
load('gnd_session3.mat');
[~,ntt] = size(fea);
XU = fea;

YU_gnd = gnd;
clear fea gnd

n = ntr + ntt;

X = [XL, XU]; 

X = X - repmat(mean(X,2),[1,ntr+ntt]);
X = mapminmax(X, -1, 1);
X = NormalizeFea(X');
X = X';
XL = X(:,1:ntr);
XU = X(:,ntr+1:n);
XL = XL - repmat(mean(XL,2),[1,ntr]);
XU = XU - repmat(mean(XU,2),[1,ntt]);


% paramters
slib = c-1;
lambdalib = 2.^(-20:10);
% lambdalib = 2.^(-4);

acc_S3LRR = zeros(length(slib),length(lambdalib));

for i = 1:length(slib)
    s = slib(i);
    for j = 1:length(lambdalib)
        lambda = lambdalib(j);
       
        
        % function
        [Y,A,B,obj] = S3LRR13v2(XL,XU,YL,s,lambda);
        YU = Y(ntr+1:n,:);
        
        % accuracy
        [~,YU_pred] = max(YU, [],2);
        acc_S3LRR(i,j) = length(find(YU_gnd == YU_pred))./ntt;
        fprintf('s = %d,lambda = %d,accuracy = %.2f\n',s,log2(lambda),acc_S3LRR(i,j)*100);
        
    end
end
 fprintf('The max accuracy = %.2f\n',max(acc_S3LRR(:))*100);
 
% analysis on 'frequency band' and 'channel'
% W = A*B;
% Wi2 = sqrt(sum(W.*W,2) + eps);
% i_delta = sum(Wi2(1:62));
% i_theta = sum(Wi2(62:124));
% i_alpha = sum(Wi2(125:186));
% i_beta  = sum(Wi2(187:248));
% i_gamma = sum(Wi2(249:310));
% figure;
% bar([i_delta,i_theta,i_alpha,i_beta,i_gamma]./sum(Wi2));
% 
% save A_1_3 A
% save B_1_3 B
% 
% save YU_pred_1_3_subject13 YU_pred