clear all
close all
clc
warning off
addpath D:\work\matlab_code\Brain-like-Retinex-main\
addpath D:\work\matlab_code\image-enhancement-about-Retinex-master\
addpath D:\work\matlab_code\LIME-master\
addpath D:\work\matlab_code\Low-light-image-enhancement-master\
addpath D:\work\matlab_code\MSRetinex\
addpath D:\work\matlab_code\Retinex-Based-Multiphase-Algorithm-for-Low-Light-Image-Enhancement-main\
addpath D:\work\matlab_code\STAR-TIP2020-master\
addpath D:\work\matlab_code\Function\
addpath D:\work\matlab_code\Brain-like-Retinex-main\image\
addpath D:\work\matlab_code\benchmark\
addpath D:\work\matlab_code\benchmark\VIFutils\
addpath D:\work\matlab_code\benchmark\VIFutils\matlabPyrTools-master\
addpath D:\work\matlab_code\benchmark\VIFutils\matlabPyrTools-master\TUTORIALS\
addpath D:\work\matlab_code\benchmark\VIFutils\matlabPyrTools-master\MEX\
addpath D:\work\matlab_code\WVM\
addpath D:\work\matlab_code\BIMEF-master\
addpath D:\work\dataset\MEF\
addpath D:\work\dataset\LIME\
addpath D:\work\dataset\ExDark\
addpath D:\work\dataset\NPE\
addpath D:\work\dataset\LOLdataset\our485\low\
addpath D:\work\dataset\LOLdataset\eval15\low\
addpath D:\work\dataset\LSRW\Huawei\low\
addpath D:\work\dataset\LSRW\Nikon\low\
addpath D:\work\matlab_code\LR3M-Method-master\code\
addpath D:\work\matlab_code\LR3M-Method-master\code\lowRank\
%%
img_or=imread('2065.jpg');
% img_or=imresize(img_or,[321,481]);
% img=imnoise(img_or,'salt & pepper',0.1);
% img=imnoise(img_or,'gaussian',0,0.01);
img=imnoise(img_or,'speckle',0.1);
% img=img_or;
% img_hsv=rgb2hsv(img);
%%
img_name='2065';
% noise_type='salt';
% noise_type='gaussian';
% % noise_type='speckle';
% % noise_type='nonoise';
% noise_level='4';
%% Brain-like-Retinex-main
K=15;            % maximum iterations
gamma=2.2;
%% Proposed
alpha = 0.05;
beta = 0.05;                                                                                                                                                                         ;
ga=3;
delta=5;
[L, R, N,out] = out_improved5(img, alpha, beta,ga,delta, K,gamma);

figure
imshow(img)
figure
imshow(L)
figure
imshow(R)
figure
imshow(N)
figure
imshow(out)
% figure
% imshow(out-N)

%% benchmark
% [ab1,de1,eme1,loe1,pixDist1,vif1,vld1] = benchmark(img_or,uint8(out2));
% [ab2,de2,eme2,loe2,pixDist2,vif2,vld2] = benchmark(img_or,uint8(out5));
% [ab3,de3,eme3,loe3,pixDist3,vif3,vld3] = benchmark(img_or,uint8(out4));
% [ab4,de4,eme4,loe4,pixDist4,vif4,vld4] = benchmark(img_or,uint8(out6));
% [ab5,de5,eme5,loe5,pixDist5,vif5,vld5] = benchmark(img_or,uint8(out7));
% [ab6,de6,eme6,loe6,pixDist6,vif6,vld6] = benchmark(img_or,uint8(out3));
% [ab7,de7,eme7,loe7,pixDist7,vif7,vld7] = benchmark(img_or,uint8(out1));
% [ab8,de8,eme8,loe8,pixDist8,vif8,vld8] = benchmark(img_or,uint8(out_ours));
% ab=[ab1,ab2,ab3,ab4,ab5,ab6,ab7,ab8];
% de=[de1,de2,de3,de4,de5,de6,de7,de8];
% eme=[eme1,eme2,eme3,eme4,eme5,eme6,eme7,eme8];
% loe=[loe1,loe2,loe3,loe4,loe5,loe6,loe7,loe8];
% pixDist=[pixDist1,pixDist2,pixDist3,pixDist4,pixDist5,pixDist6,pixDist7,pixDist8];
% vif=[vif1,vif2,vif3,vif4,vif5,vif6,vif7,vif8];
% vld=[vld1,vld2,vld3,vld4,vld5,vld6,vld7,vld8];
% benchmark_final=[ab;de;eme;loe;pixDist;vif;vld]';
%% benchmark1
% result1=PSNR(img_or,out2);
% result2=PSNR(img_or,out5);
% result3=PSNR(img_or,out4);
% result4=PSNR(img_or,out6);
% result5=PSNR(img_or,out7);
% result6=PSNR(img_or,out3);
% result7=PSNR(img_or,out1);
% result8=PSNR(img_or,out_ours);
% result=[result1,result2,result3,result4,result5,result6,result7,result8];