function [L, R,N,eIm,C,G] = out_improved5(src, alpha, beta,ga,delta, K,gamma)
%% process three channel
c=size(src,3);
L=zeros(size(src));
R=zeros(size(src));
N=zeros(size(src));
C=zeros(size(src));
G=zeros(size(src));
src=im2double(src);
for i=1:c
%     [L(:,:,i), R(:,:,i),N(:,:,i)] = noise_algorithm(src(:,:,i), alpha, beta,ga,delta, K);
    [L(:,:,i), R(:,:,i),N(:,:,i),C(:,:,i),G(:,:,i)] = new_noise_algorithm(src(:,:,i), alpha, beta,ga,delta, K);
end
eIm=(R).*L.^(1/gamma);
% for i=1:3
%     Nabs = abs(N(:,:,i));
%     t = 3 * median(Nabs(:));        % 或者 prctile(Nabs(:), 95)
%     mask = Nabs > t;                % 噪点区域
%     mask = bwareaopen(mask, 5);     % 去掉零碎小点（可选）
%     eIm(:,:,i) = regionfill(eIm(:,:,i), mask);
% end
end

