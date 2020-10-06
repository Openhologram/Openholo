clc;
fclose('all');

% unit
mm = 1e-3;
um = mm*mm;
nm = um*mm;

% inline command
%asm_kernel = @(f, wvl, x, y, res, pp) exp(-2j*pi*f .* sqrt( wvl^-2 - ((x/pp - 0.5)./pp/res).^2 - ((y/pp - 0.5)./pp/res).^2));     % angular spectrum method kernel
norm2int = @(img) uint8(255.*img./max(max(img)));

% parameters
h = 2160; v = 3840;
wvl = 525 * nm;  % green
pp = 3.6*um; % pixel pitch
res = v/2;
ReconstLength = 1;

% input file name

fim_py = 'File name_IM.bmp';
fre_py = 'File name_RE.bmp';

%% Convert to complex image
% phase part
im_py = imread(fim_py);
im_py = double(im_py) ./ 255;
im_py = im_py(:, (v-h)/2+1:(v+h)/2, 2);  % crop
phase = (im_py - 0.5) .* (2*pi);

% amplitude part
re_py = imread(fre_py);
re_py = double(re_py) ./ 255;
re_py = re_py(:,(v-h)/2+1:(v+h)/2, 2);

% adapt ASM
r = (-pp*h/2 + pp/2):pp:(pp*h/2 - pp/2);
c =  r; 
[C, R] = meshgrid(c, r);

% Complex image
%ch = real + 1j*imag;
ch = re_py .* exp(1j.*phase);
A = fftshift(fft2(fftshift(ch)));

p = asm_kernel(zz, wvl, C, R, h, pp);
%% Reconstruction part

figure;
for zz = (400:10:980)*mm
%for zz = ReconstLength * mm
    p = asm_kernel(zz, wvl, C, R, h, pp);
    Az1 = A .* p;
    EI = fftshift(ifft2(fftshift(Az1)));
    I_rec = EI .* conj(EI);
    I_rec = I_rec / max(max(I_rec)); %normalize term
    I_rec = 255 .* I_rec;
    I_rec = uint8(I_rec);
    %imwrite(I_rec, sprintf('OPH_point_3_%.2fmm.jpg', zz/mm+20));
    fig = imagesc(I_rec);
    title(sprintf('ASM 3 point Reconst %.2f mm', zz/mm));
    %saveas(fig, sprintf('ASM 3 point Reconst %.2f mm.jpg', zz/mm));
    pause(0.2);
end