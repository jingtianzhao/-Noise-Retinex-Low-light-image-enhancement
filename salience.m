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