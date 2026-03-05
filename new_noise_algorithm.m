function [L, R, N,Contour0,DS_S] = new_noise_algorithm(src, alpha, beta, chi, delta, K)
S = double(src);
[Contour0, ~] = salience(src);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
C_Idx = (Contour0 > 0.1 * max(Contour0(:)));

N_row = 3; sigma = 1;
gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
center = imfilter(S, gausFilter, 'symmetric');

N_row = 7; sigma = 3;
gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
surround = imfilter(S, gausFilter, 'symmetric');

edge = center - 0.7 * surround;
edge(abs(edge) < 0.01 * max(abs(edge(:)))) = 0;

Contour = edge .* C_Idx;        % C
DS_S    = edge - Contour;       % G
% figure
% imshow(DS_S)
% figure
% imshow(Contour0)

blur1 = imgaussfilt(S, 1);
blur2 = imgaussfilt(S, 8);
blur2(blur2 < 1e-6) = 1e-6;

L = blur2;
R = 0.75 * (blur1 ./ blur2);
N = zeros(size(S));

Sh = S - medfilt2(S, [3 3], 'symmetric');
eps0 = 1e-6;

for iter = 1:K

    DR = imgaussfilt(R,1) - imgaussfilt(R,3);
    DL = imgaussfilt(L,1) - imgaussfilt(L,3);
    DN = imgaussfilt(N,1) - imgaussfilt(N,3);

    R = ( ...
        L .* (S - L .* N) ...
        + alpha * DL .* Contour ...
        + beta  * (L .* DR) .* DS_S ...
        ) ./ ( ...
        L.^2 ...
        + alpha * DL.^2 ...
        + beta  * (L .* DR).^2 ...
        + eps0 );

    R = max(R, 1e-4);
%     R=R-N;
%     Nabs = abs(N);
%     t = 3 * median(Nabs(:));        % 或者 prctile(Nabs(:), 95)
%     mask = Nabs > t;                % 噪点区域
%     mask = bwareaopen(mask, 5);     % 去掉零碎小点（可选）
%     R = regionfill(R, mask);
%     % 强制 R 与 N 低相关（去掉噪声方向）
%     proj = (R(:)' * N(:)) / (N(:)' * N(:) + eps);
%     R= R - proj * N;

    L = ( ...
        (R + N) .* S ...
        + beta * (DR .* R) .* DS_S ...
        + chi  * (DN + N .* DL) .* (DL .* N) ...
        ) ./ ( ...
        (R + N).^2 ...
        + beta * (DR .* R).^2 ...
        + chi  * (DN + N .* DL).^2 ...
        + eps0 );


    L(L < 1e-3) = 1e-3;

    N = ( ...
        L .* (S - L .* R) ...
        + chi * (DN .* L + DL) .* (DL .* N) ...
        + delta * Sh ...
        ) ./ ( ...
        L.^2 ...
        + chi * (DN .* L + DL).^2 ...
        + delta ...
        + eps0 );

end
     L = imgaussfilt(L, 2);   % sigma=2 或 3
    

