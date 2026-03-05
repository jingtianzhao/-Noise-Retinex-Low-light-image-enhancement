function [L, R] = RlDeC_main(src, alpha, beta, K, vareps)
   
    src=im2double(src);
    if (~exist('alpha', 'var'))
        alpha = 0.001;
    end
    if (~exist('beta', 'var'))
        beta = 0.0001;
    end

    if (~exist('vareps', 'var'))
        vareps = 0.001;
    end

    if (~exist('K', 'var'))
        K = 20;
    end

    fprintf('-- Stop iteration until eplison < %02f or K > %d\n', vareps, K);

    if size(src, 3) == 1
        S = src;
    else
        hsv = rgb2hsv(src);
        S = hsv(:, :, 3);
    end

    disp('Start..............')
    [Contour, ~] = salience(src);
    C_Idx = (Contour > 0.1 * max(max(Contour)));
    N_row = 3;
    sigma = 1;
    gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
    center = imfilter(S, gausFilter, 'symmetric');
    N_row = 7;
    sigma = 3;
    gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
    surround = imfilter(S, gausFilter, 'symmetric');
    edge = 1 * (1 * center - 0.7 * surround);
    e_Inx = (edge > 0.01 * max(max(edge)));
    edge = edge .* e_Inx;
    Contour = edge .* C_Idx; %ÂÖÀª
    edge = edge - Contour;
    DS_S = edge; %ÎÆÀí
    N_row = 3;
    sigma = 1;
    gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
    blur1 = imfilter(S, gausFilter, 'symmetric');
    N_row = 40;
    sigma = 8;
    gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
    blur2 = imfilter(S, gausFilter, 'symmetric');
    blur2(blur2 < 0.000000001) = 0.000000001;
    L = blur2;
    R = 0.75 * (blur1 ./ blur2);
    for iter = 1:K
        preL = L;
        preR = R;
        N_row = 3;
        sigma = 1;
        gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
        center = imfilter(R, gausFilter, 'symmetric');
        N_row = 7;
        sigma = 3;
        gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
        surround = imfilter(R, gausFilter, 'symmetric');
        surround(surround < 0.0001) = 0.0001;
        edge = 1 * (1 * center - 0.7 * surround) ./ surround;
        e_Inx = (edge > 0.01 * max(max(edge)));
        edge = edge .* e_Inx;
        %% DR
        DR = edge;
        N_row = 3;
        sigma = 1;
        gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
        center = imfilter(L, gausFilter, 'symmetric');
        N_row = 7;
        sigma = 3;
        gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
        surround = imfilter(L, gausFilter, 'symmetric');
        edge = 1 * (1 * center - 0.7 * surround);
        e_Inx = (edge > 0.01 * max(max(edge)));
        edge = edge .* e_Inx;
        DL = edge;
        %% L ºÍDL
        L = SLSl(S, R, DL, DR, Contour, DS_S, alpha, beta);
        eplisonL = norm(L - preL, 'fro') / norm(preL, 'fro');
        N_row = 3;
        sigma = 1;
        gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
        center = imfilter(L, gausFilter, 'symmetric');
        N_row = 7;
        sigma = 3;
        gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
        surround = imfilter(L, gausFilter, 'symmetric');
        edge = 1 * (1 * center - 0.7 * surround);
        e_Inx = (edge > 0.01 * max(max(edge)));
        edge = edge .* e_Inx;
        DL = edge;
        %% R
        R = SLSr(S, L, DL, DR, Contour, DS_S, alpha, beta);
        eplisonR = norm(R - preR, 'fro') / norm(preR, 'fro');
        fprintf('Iter #%d : eplisonL = %f; eplisonR = %f\n', iter, eplisonL, eplisonR);

        if (eplisonL < vareps && eplisonR < vareps)
            break;
        end

    end

    disp('Done..............')
    L(L < 0) = 0;
    R(R < 0) = 0;
end

function dst = SLSl(s, lr, xy, DeR, t, G, alpha, beta)

    if (~exist('method', 'var'))
        method = 'pcg';
    end

    [h, w] = size(s);
    hw = h * w;
    aa = 0.0001;
    te = ones(size(s)) * aa;
    E = spdiags(te(:), 0, hw, hw);
    D = spdiags(xy(:), 0, hw, hw);
    Dk = spdiags(lr(:), 0, hw, hw);
    Dtrk = spdiags(DeR(:), 0, hw, hw);
    A = Dk' * Dk + beta * (Dtrk' * Dtrk) + E;
    b = Dk' * s(:) + beta * Dtrk' * G(:) + 0.000001;

    switch method
        case 'pcg'
            L = ichol(A, struct('michol', 'on'));
            [dst, ~] = pcg(A, b, 0.01, 40, L, L');
        case 'minres'
            [dst, ~] = minres(A, b, 0.01, 40);
        case 'bicg'
            [L, U] = ilu(A, struct('type', 'ilutp', 'droptol', 0.01));
            [dst, ~] = bicg(A, b, 0.01, 40, L, U);
        case 'direct'
            [dst, ~] = A \ b; %#ok
    end

    dst = reshape(dst, h, w);
end

function dst = SLSr(s, lr, DeL, xy, C, t, alpha, beta)

    if (~exist('method', 'var'))
        method = 'pcg';
    end

    [h, w] = size(s);
    hw = h * w;
    aa = 0.0001;
    te = ones(size(s)) * aa;
    E = spdiags(te(:), 0, hw, hw);
    D = spdiags(xy(:), 0, hw, hw);
    Dk = spdiags(lr(:), 0, hw, hw);
    Dtlk = spdiags(DeL(:), 0, hw, hw);
    A = Dk' * Dk + alpha * (Dtlk' * Dtlk) + E;
    b = Dk' * s(:) + alpha * Dtlk' * C(:) + 0.000001;

    switch method
        case 'pcg'
            L = ichol(A, struct('michol', 'on'));
            [dst, ~] = pcg(A, b, 0.01, 40, L, L');
        case 'minres'
            [dst, ~] = minres(A, b, 0.01, 40);
        case 'bicg'
            [L, U] = ilu(A, struct('type', 'ilutp', 'droptol', 0.01));
            [dst, ~] = bicg(A, b, 0.01, 40, L, U);
        case 'direct'
            [dst, ~] = A \ b; %#ok
    end

    dst = reshape(dst, h, w);
end

function [R_NCRF, resp] = salience(rgb)
    InputImage = imresize(rgb, 1);

    if max(InputImage(:)) > 1
        InputImage = double(InputImage);
        InputImage = InputImage ./ max(InputImage(:));
    end

    LgnSigma = 1.1;
    OpponentImage = SingleOpponent(InputImage, LgnSigma);
    [h, w, s] = size(OpponentImage);
    type = 'DOG';
    sigma_y = [4];
    WR = 6;
    AR = [1 * WR];
    ndir = 12;

    for i = 1:s
        originimg = reshape(OpponentImage(:, :, i), [h, w]);
        maxs = Boundarynon(originimg, sigma_y, WR, AR, type);
        maxs = maxs ./ max(maxs(:));
        outm(:, :, :, :, i) = maxs(:, :, :, :);
    end

    [R_NCRF, Ori] = Process_V1SS(outm, ndir);
    resp = colnon(R_NCRF, Ori, ndir, 1);
end

function outm = Boundarynon(img, sigma_y, WR, AR, type)
    ndir = 12;
    for i = 1:length(sigma_y)
        f(i) = struct('sigma_y', sigma_y(i), 'size', 4 * sigma_y(i) + 1,'sigma_x1', sigma_y(i) / AR(i), 'sigma_x2', WR * sigma_y(i) / AR(i),'pr', 1.5, 'x0', 0, 'y0', 0);
    end
    t = type;
    coeff = pi / ndir;
    V1 = struct('max', zeros(size(img, 1), size(img, 2)),'angles', zeros(size(img, 1), size(img, 2)),'simple_e', zeros(ndir, length(f), size(img, 1), size(img, 2)),'simple_o', zeros(ndir, length(f), size(img, 1), size(img, 2)),'complex_e', zeros(ndir, length(f), size(img, 1), size(img, 2)),'complex_o', zeros(ndir, length(f), size(img, 1), size(img, 2)));
    i = 1;
    p =- 2:2;
    w = 1 ./ (sqrt(2 * pi)) .* exp(- (p) .^ 2 ./ 2) * 2.5;

    for ii = 1:ndir
        thetas(ii) = (ii - 1) * pi / ndir;
    end

    for th = 0:coeff:pi - coeff
        theta = thetas(i);

        for j = 1:length(f)

            if strcmp(t, 'Gabor')
                [fo, fe] = GaborD(f(j). size, f(j). sigma_y, f(j). sigma_x1, theta, f(j). pr, f(j). x0, f(j). y0);
            else
                [fo, fe] = DOG(f(j). size, f(j). sigma_y, f(j). sigma_x1, f(j). sigma_x2, theta, f(j). x0, f(j). y0);
                [fe0, fo0] = GaborD(5, 1, 1, theta, 0, 0, 0);
            end

            simple = abs(imfilter(double(img), fe, 'symmetric'));
            V1. simple_e(i, j, :, :) = abs(imfilter(simple, fe0, 'symmetric'));
            se = 2 * (f(j). sigma_y / f(j). sigma_x1);
            a = int16((f(j). x0 + floor(f(j). size / se)) * sin(mod(th + pi / 2, pi)));
            b = int16((f(j). y0 + floor(f(j). size / se)) * cos(mod(th + pi / 2, pi)));
            auxe = V1. simple_e(i, j, :, :);
            auxo = V1. simple_o(i, j, :, :);
            auxe(auxe < 0) = 0;
            auxo(auxo < 0) = 0;

            if (b < 0)
                b = abs(b);
                V1. complex_e(i, j, :, :) = auxe(1, 1, :, :) * w(3);
                V1. complex_e(i, j, 1:size(auxe, 3) - a, 1:size(auxe, 4) - b) = V1. complex_e(i, j, 1:size(auxe, 3) - a, 1:size(auxe, 4) - b) + auxe(1, 1, a + 1:size(auxe, 3), b + 1:size(auxe, 4)) * w(2);
                V1. complex_e(i, j, 1:size(auxe, 3) - 2 * a, 1:size(auxe, 4) - 2 * b) = V1. complex_e(i, j, 1:size(auxe, 3) - 2 * a, 1:size(auxe, 4) - 2 * b) + auxe(1, 1, 2 * a + 1:size(auxe, 3), 2 * b + 1:size(auxe, 4)) * w(1);
                V1. complex_e(i, j, a + 1:size(auxe, 3), b + 1:size(auxe, 4)) = V1. complex_e(i, j, a + 1:size(auxe, 3), b + 1:size(auxe, 4)) + auxe(1, 1, 1:size(auxe, 3) - a, 1:size(auxe, 4) - b) * w(4);
                V1. complex_e(i, j, 2 * a + 1:size(auxe, 3), 2 * b + 1:size(auxe, 4)) = V1. complex_e(i, j, 2 * a + 1:size(auxe, 3), 2 * b + 1:size(auxe, 4)) + auxe(1, 1, 1:size(auxe, 3) - 2 * a, 1:size(auxe, 4) - 2 * b) * w(5);
            else
                V1. complex_e(i, j, :, :) = auxe(1, 1, :, :) * w(3);
                V1. complex_e(i, j, 1:size(auxe, 3) - a, 4 * b + 1:size(auxe, 4)) = V1. complex_e(i, j, 1:size(auxe, 3) - a, 4 * b + 1:size(auxe, 4)) + auxe(1, 1, a + 1:size(auxe, 3), 3 * b + 1:size(auxe, 4) - b) * w(2);
                V1. complex_e(i, j, 1:size(auxe, 3) - 2 * a, 4 * b + 1:size(auxe, 4)) = V1. complex_e(i, j, 1:size(auxe, 3) - 2 * a, 4 * b + 1:size(auxe, 4)) + auxe(1, 1, 2 * a + 1:size(auxe, 3), 2 * b + 1:size(auxe, 4) - 2 * b) * w(1);
                V1. complex_e(i, j, a + 1:size(auxe, 3), 3 * b + 1:size(auxe, 4) - b) = V1. complex_e(i, j, a + 1:size(auxe, 3), 3 * b + 1:size(auxe, 4) - b) + auxe(1, 1, 1:size(auxe, 3) - a, 4 * b + 1:size(auxe, 4)) * w(4);
                V1. complex_e(i, j, 2 * a + 1:size(auxe, 3), 2 * b + 1:size(auxe, 4) - 2 * b) = V1. complex_e(i, j, 2 * a + 1:size(auxe, 3), 2 * b + 1:size(auxe, 4) - 2 * b) + auxe(1, 1, 1:size(auxe, 3) - 2 * a, 4 * b + 1:size(auxe, 4)) * w(5);
            end

        end

        i = i + 1;
    end

    outm = V1. simple_e;
end

function SMat = CentreZero(SMat, CRadius)
    [h, w] = size(SMat);

    if CRadius == 0 || CRadius > min(h, w)
        return;
    end

    x = max(h, w);

    if mod(x, 2) == 0
        x = x + 1;
    end

    [hh, ww] = meshgrid(1:x);
    centre = ceil(x / 2);
    ch = sqrt((hh - centre) .^ 2 + (ww - centre) .^ 2) <= CRadius;
    SMat(ch) = 0;
end

function [Resout, Orout] = ChannelMax(Resinput, Orinput)
    [h, w, ~] = size(Resinput);
    Orout = zeros(h, w);
    SumResponse = sum(Resinput, 3);
    [~, MaxInds] = max(Resinput, [], 3);
    Resout = SumResponse;

    if ~isempty(Orinput)

        for c = 1:max(MaxInds(:))
            corien = Orinput(:, :, c);
            Orout(MaxInds == c) = corien(MaxInds == c);
        end

    end

end

function resp = colnon(EdgeImageResponse, FinalOrientations, dir, flag)
    EdgeImageResponse = EdgeImageResponse ./ max(EdgeImageResponse(:));

    if flag == 1
        FinalOrientations = (FinalOrientations - 1) * pi / dir;
        FinalOrientations = mod(FinalOrientations + pi / 2, pi);
    end

    EdgeImageResponse = NonMaxChannel(EdgeImageResponse, FinalOrientations);
    EdgeImageResponse([1, end], :) = 0;
    EdgeImageResponse(:, [1, end]) = 0;
    resp = EdgeImageResponse ./ max(EdgeImageResponse(:));
end

function [dog, doog] = DOG(size, sigma_y, sigma_x1, sigma_x2, theta, x0, y0)
    [X, Y] = meshgrid(- fix(size / 2):fix(size / 2), fix(- size / 2):fix(size / 2));
    x_theta = (X - x0) * cos(theta) + (Y - y0) * sin(theta);
    y_theta =- (X - x0) * sin(theta) + (Y - y0) * cos(theta);
    dog1 = 1 / (2 * pi * sigma_x1 * sigma_y) .* exp(- .5 * (x_theta .^ 2 / sigma_x1 ^ 2 + y_theta .^ 2 / sigma_y ^ 2));
    dog2 = 1 / (2 * pi * sigma_x2 * sigma_y) .* exp(- .5 * (x_theta .^ 2 / sigma_x2 ^ 2 + y_theta .^ 2 / sigma_y ^ 2));
    dog = dog1 - dog2;
    x1 = X .* cos(theta) + Y .* sin(theta);
    y1 =- X .* sin(theta) + Y .* cos(theta);
    doog =- x1 .* exp(- ((x1 .^ 2) * sigma_x2 + (y1 .^ 2) * sigma_x1) / (2 * (sigma_y ^ 2))) / (pi * (sigma_y ^ 2));
end

function [G_even, G_odd] = GaborD(n, sigma_y, sigma_x, theta, pr, x0, y0)

    if length(n) > 1,
        incx = 2 * n / (n(1) - 1);
        incy = 2 * n / (n(2) - 1);
    else
        incx = 2 * n / (n - 1);
        incy = 2 * n / (n - 1);
    end

    [X, Y] = meshgrid(- n:incx:n, - n:incy:n);
    xp = (X - x0) * cos(theta) + (Y - y0) * sin(theta);
    yp =- (X - x0) * sin(theta) + (Y - y0) * cos(theta);
    G_even = 1 ./ (2 * pi * sigma_x * sigma_y) .* exp(- .5 * (xp .^ 2 ./ sigma_x ^ 2 + yp .^ 2 ./ sigma_y ^ 2)) .*cos(2 * pi / (4 * sigma_x) * pr .* xp);
    G_odd = 1 ./ (2 * pi * sigma_x * sigma_y) .* exp(- .5 * ((xp - x0) .^ 2 ./ sigma_x ^ 2 + (yp - y0) .^ 2 ./ sigma_y ^ 2)) .*sin(2 * pi / (4 * sigma_x) * pr .* xp);
end

function g = gaus2D(sgm)
    nr = 15 * sgm;
    nc = 15 * sgm;
    [x, y] = meshgrid(linspace(- nc / 2, nc / 2, nc), linspace(nr / 2, - nr / 2, nr));
    g = exp(- (x .^ 2 + y .^ 2) / (2 * (sgm) ^ 2));
    g = g / sum(g(:));
end

function h = GaussianFilter2(sigmax, sigmay, meanx, meany, theta)

    if nargin < 1 || isempty(sigmax)
        sigmax = 0.5;
    end

    if nargin < 2 || isempty(sigmay)
        sigmay = sigmax;
    end

    if sigmax == 0 || sigmay == 0
        h = 1;
        return;
    end

    MaxSigma = max(sigmax, sigmay);
    sizex = Gausswidth(MaxSigma);
    sizey = sizex;

    if nargin < 3 || isempty(meanx)
        meanx = 0;
    end

    if nargin < 4 || isempty(meany)
        meany = 0;
    end

    if nargin < 5 || isempty(theta)
        theta = 0;
    end

    centrex = (sizex + 1) / 2;
    centrey = (sizey + 1) / 2;
    centrex = centrex + (meanx * centrex);
    centrey = centrey + (meany * centrey);
    xs = linspace(1, sizex, sizex)' * ones(1, sizey) - centrex;
    ys = ones(1, sizex)' * linspace(1, sizey, sizey) - centrey;
    a = cos(theta) ^ 2/2 / sigmax ^ 2 + sin(theta) ^ 2/2 / sigmay ^ 2;
    b =- sin(2 * theta) / 4 / sigmax ^ 2 + sin(2 * theta) / 4 / sigmay ^ 2;
    c = sin(theta) ^ 2/2 / sigmax ^ 2 + cos(theta) ^ 2/2 / sigmay ^ 2;
    h = exp(- (a * xs .^ 2 + 2 * b * xs .* ys + c * ys .^ 2));
    h = h ./ sum(h(:));
end

function FilterWidth = Gausswidth(sigma, MaxWidth)

    if nargin < 2
        MaxWidth = 100;
    end

    threshold = 1e-4;
    n = 1:MaxWidth;
    FilterWidth = find(exp(- (n .^ 2) / (2 * sigma .^ 2)) > threshold, 1, 'last');

    if isempty(FilterWidth)
        FilterWidth = 1;
    end

    FilterWidth = FilterWidth .* 2 + 1;
end

function ch = LocalAverage(Radius)
    x = Radius * 2;

    if mod(x, 2) == 0
        x = x + 1;
    end

    [h, w] = meshgrid(1:x);
    centre = ceil(x / 2);
    ch = sqrt((h - centre) .^ 2 + (w - centre) .^ 2) <= Radius;
    ch = ch ./ sum(ch(:));
end

function ImageStd = LocalStd(InputImage, SRadius, CRadius)
    InputImage = double(InputImage);

    if nargin < 2
        SRadius = 2.5;
    end

    if nargin < 3
        CRadius = 0;
    end

    Temf = LocalAverage(SRadius);
    Temf = CentreZero(Temf, CRadius);
    Temf = Temf ./ sum(Temf(:));
    MeanCentre = imfilter(InputImage, Temf, 'symmetric');
    stdv = (InputImage - MeanCentre) .^ 2;
    ImageStd = sqrt(imfilter(stdv, Temf, 'symmetric'));
end

function d = NonMaxChannel(d, t)

    for i = 1:size(d, 3)
        d(:, :, i) = d(:, :, i) ./ max(max(d(:, :, i)));
        d(:, :, i) = nonmax(d(:, :, i), t(:, :, i));
        d(:, :, i) = max(0, min(1, d(:, :, i)));
    end

end

function noarmalisedx = NormaliseChannel(x, a, b, mins, maxs)

    if nargin < 2
        a = [];
        b = [];
        mins = [];
        maxs = [];
    end

    [rows, cols, chns] = size(x);
    OriginalMin = min(x(:));
    OriginalMax = max(x(:));

    if isempty(a)

        if OriginalMin >= 0
            a = 0.0;
        else
            a =- 1.0;
        end

    end

    if isempty(b)

        if OriginalMax >= 0
            b = 1.0;
        else
            b = 0.0;
        end

    end

    if length(a) == 1
        a(2:chns) = a(1);
    end

    if length(b) == 1
        b(2:chns) = b(1);
    end

    if chns == 1 && cols == 3
        x = reshape(x, rows, cols / 3, 3);
    end

    x = double(x);
    a = double(a);
    b = double(b);
    noarmalisedx = zeros(size(x));

    if isempty(mins)
        mins = min(x(:));
    end

    if isempty(maxs)
        maxs = max(x(:));
    end

    if length(mins) == 1
        mins(2:chns) = mins(1);
    end

    if length(maxs) == 1
        maxs(2:chns) = maxs(1);
    end

    for i = 1:chns
        minv = mins(i);
        maxv = maxs(i);

        if a(i) == b(i)
            noarmalisedx(:, :, i) = a(i);
        else
            noarmalisedx(:, :, i) = a(i) + (x(:, :, i) - minv) * (b(i) - a(i)) / (maxv - minv);
        end

    end

    if chns == 1
        noarmalisedx = reshape(noarmalisedx, rows, cols);
    end

end

function dblImageSX = normrange(grayImage, desiredMin, desiredMax)
    originalMinValue = double(min(min(grayImage)));
    originalMaxValue = double(max(max(grayImage)));
    originalRange = originalMaxValue - originalMinValue;
    desiredRange = desiredMax - desiredMin;
    dblImageSX = desiredRange * (double(grayImage) - originalMinValue) / originalRange + desiredMin;
end

function [im] = nonmax(im, theta)

    if numel(theta) == 1,
        theta = theta .* ones(size(im));
    end

    theta = mod(theta + pi / 2, pi);
    mask15 = (theta >= 0 & theta < pi / 4);
    mask26 = (theta >= pi / 4 & theta < pi / 2);
    mask37 = (theta >= pi / 2 & theta < pi * 3/4);
    mask48 = (theta >= pi * 3/4 & theta < pi);
    mask = ones(size(im));
    [h, w] = size(im);
    [ix, iy] = meshgrid(1:w, 1:h);
    idx = find(mask15 & ix < w & iy < h);
    idxA = idx + h;
    idxB = idx + h + 1;
    d = tan(theta(idx));
    imI = im(idxA) .* (1 - d) + im(idxB) .* d;
    mask(idx(find(im(idx) <= imI))) = 0;
    idx = find(mask15 & ix > 1 & iy > 1);
    idxA = idx - h;
    idxB = idx - h - 1;
    d = tan(theta(idx));
    imI = im(idxA) .* (1 - d) + im(idxB) .* d;
    mask(idx(find(im(idx) <= imI))) = 0;
    idx = find(mask26 & ix < w & iy < h);
    idxA = idx + 1;
    idxB = idx + h + 1;
    d = tan(pi / 2 - theta(idx));
    imI = im(idxA) .* (1 - d) + im(idxB) .* d;
    mask(idx(find(im(idx) <= imI))) = 0;
    idx = find(mask26 & ix > 1 & iy > 1);
    idxA = idx - 1;
    idxB = idx - h - 1;
    d = tan(pi / 2 - theta(idx));
    imI = im(idxA) .* (1 - d) + im(idxB) .* d;
    mask(idx(find(im(idx) <= imI))) = 0;
    idx = find(mask37 & ix > 1 & iy < h);
    idxA = idx + 1;
    idxB = idx - h + 1;
    d = tan(theta(idx) - pi / 2);
    imI = im(idxA) .* (1 - d) + im(idxB) .* d;
    mask(idx(find(im(idx) <= imI))) = 0;
    idx = find(mask37 & ix < w & iy > 1);
    idxA = idx - 1;
    idxB = idx + h - 1;
    d = tan(theta(idx) - pi / 2);
    imI = im(idxA) .* (1 - d) + im(idxB) .* d;
    mask(idx(find(im(idx) <= imI))) = 0;
    idx = find(mask48 & ix > 1 & iy < h);
    idxA = idx - h;
    idxB = idx - h + 1;
    d = tan(pi - theta(idx));
    imI = im(idxA) .* (1 - d) + im(idxB) .* d;
    mask(idx(find(im(idx) <= imI))) = 0;
    idx = find(mask48 & ix < w & iy > 1);
    idxA = idx + h;
    idxB = idx + h - 1;
    d = tan(pi - theta(idx));
    imI = im(idxA) .* (1 - d) + im(idxB) .* d;
    mask(idx(find(im(idx) <= imI))) = 0;
    im = im .* mask;
end

function [TT, Tt] = Process_V1SS(outm, nThetas)
    sigmax = 1;
    sigmay = sigmax / 8;
    Surroundsize = 2;
    [Rest, Ort] = max(outm, [], 1);
    Rests(:, :, :) = Rest(1, 1, :, :, :);
    Orts(:, :, :) = Ort(1, 1, :, :, :);

    for c = 1:size(Rests, 3)
        P_Channel = Rests(:, :, c);
        P_Orientation = Orts(:, :, c);
        ss = LocalStd(P_Channel, 45/2);
        ss = ss ./ max(ss(:));
        ss = max(ss(:)) - ss;
        ss = NormaliseChannel(ss, 0.7, 1.0, [], []);

        for t = 1:nThetas
            theta = (t - 1) * pi / nThetas;
            theta = theta + (pi / 2);
            responsec = imfilter(Rests(:, :, c), GaussianFilter2(sigmax, sigmay, 0, 0, theta), 'symmetric');
            responses = imfilter(Rests(:, :, c), GaussianFilter2(sigmax * Surroundsize, sigmay * Surroundsize, 0, 0, theta), 'symmetric');
            response = max(responsec - ss .* responses, 0);
            P_Channel(P_Orientation == t) = response(P_Orientation == t);
        end

        Rests(:, :, c) = P_Channel;
    end

    [TT, Tt] = ChannelMax(Rests, Orts);
end

function [Bcon, Bcoff, OPL, OUT] = Retina_no_temporal(originimg, alpha_ph, alpha_h)
    dblImageSX = normrange(originimg, 0, 1);
    hh = 10;
    ww = 10;
    a1 =- 0.5:1 / hh:0.5 - 1 / hh;
    b1 =- 0.5:1 / ww:0.5 - 1 / ww;
    fk1 = repmat(a1', 1, ww);
    fk2 = repmat(b1, hh, 1);
    beta_ph = 0;
    fph = 1 ./ (1 + beta_ph + 4 * alpha_ph - 2 * alpha_ph * (cos(2 * pi .* fk1)) - 2 * alpha_ph * (cos(2 * pi .* fk2)));
    beta_h = 0;
    fh = 1 ./ (1 + beta_h + 4 * alpha_h - 2 * alpha_h * (cos(2 * pi .* fk1)) - 2 * alpha_h * (cos(2 * pi .* fk2)));
    Lp = conv2(dblImageSX, fh, 'same');
    V0 = 0.7;
    Vmax = 1;
    R0 = Lp * V0 + Vmax * (1 - V0);
    Cp = (dblImageSX ./ (dblImageSX + R0)) .* (Vmax + R0);
    BPph = conv2(Cp, fph, 'same');
    BPph = normrange(BPph, 0, 1);
    BPr = conv2(Cp, fh, 'same');
    BPr = normrange(BPr, 0, 1);
    Lp = BPr;
    V00 = 0.9;
    Bcon = BPph - BPr;
    Bcon = Bcon ./ (Bcon + Lp * V00 + Vmax * (1 - V00)) .* (Vmax + Lp * V00 + Vmax * (1 - V00));
    Bcoff =- BPph + BPr;
    Bcoff = Bcoff ./ (Bcoff + Lp * V00 + Vmax * (1 - V00)) .* (Vmax + Lp * V00 + Vmax * (1 - V00));
    OPL = Bcon - Bcoff;
    OUT = OPL + Cp / 1;
end

function SOImage = SingleOpponent(InputImage, LgnSigma)
    InputImage = imfilter(InputImage, gaus2D(LgnSigma), 'conv', 'replicate');
    InputImage = sqrt(InputImage);

    if size(InputImage, 3) == 3
        [Bcon, Bcoff, OPL, OUT] = Retina_no_temporal(mean(InputImage(:, :, 1:3), 3), 7, 0.57);
        R = InputImage(:, :, 1);
        G = InputImage(:, :, 2);
        B = InputImage(:, :, 3);
        Y = (R + G) / 2;
        SOImage(:, :, 1) = R - G;
        SOImage(:, :, 2) = B - Y;
        SOImage(:, :, 3) = (mean(InputImage(:, :, 1:3), 3));
        SOImage(:, :, 4) = OUT;
        SOImage(:, :, 5) = B - 0.7 * Y;
        SOImage(:, :, 6) = sqrt(LocalStd(rgb2gray(InputImage), 1));
    else
        [Bcon, Bcoff, OPL, OUT] = Retina_no_temporal(InputImage, 7, 0.57);
        SOImage(:, :, 1) = InputImage;
        SOImage(:, :, 2) = sqrt(LocalStd(InputImage));
        SOImage(:, :, 3) = OUT;
    end

end