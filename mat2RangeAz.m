load('ants_location.mat');
load('Raw_data/Training/object_standing/objects_standing3.mat')
freqStartMHz = 62*1000;
freqStopMHz = 69*1000;
numFreqPoints = 75;
Freqs = linspace(freqStartMHz*1e6, freqStopMHz*1e6, numFreqPoints);
Nfft = 256;
tx=1:20;
rx=21:40;
TxRxPairs = zeros(length(tx)*length(rx),2);
for i=1:20
   for j=1:20
       TxRxPairs((i-1)*20+j,1) = i;
       TxRxPairs((i-1)*20+j,2) = j+20;
   end
end
tic
%Create a steering vector
numOfDigitalBeams = 32; % number of digital rx beams
% theta_vec = linspace(deg2rad(-60), deg2rad(60), numOfDigitalBeams); % Azimuth
theta_vec = asin(linspace(sind(-60), sind(60), numOfDigitalBeams)); % Azimuth
phi_s = deg2rad(0.0); % Elevation

taylor_win = taylorwin(20, 4, -40);
numOfTx = size(TxRxPairs,1)/20;
taylor_win_El = repmat(taylor_win, numOfTx, size(Freqs,2));
% reduce channel data x2
%p = randperm(20);
%taylor_win(p(1:10)) = 0;
taylor_win_Az = repmat(repelem(taylor_win, numOfTx), 1, size(Freqs,2));


% Create a steering matrix for all pairs location
H = zeros(size(TxRxPairs,1),size(Freqs,2), numOfDigitalBeams);
for beam_idx = 1:numOfDigitalBeams
    theta = theta_vec(beam_idx); % create new digital beam
    
    %K_vec_x = 2*pi*Freqs*sin(theta)/3e8;
    %K_vec_y = 2*pi*Freqs*sin(phi)/3e8;
    
    K_vec_x = Freqs*sin(theta)/3e8;
    K_vec_y = Freqs*sin(phi_s)/3e8;
    
    for ii = 1:size(TxRxPairs,1)
        D = VtrigU_ants_location(TxRxPairs(ii,1),:)+VtrigU_ants_location(TxRxPairs(ii,2),:);
        H(ii,:, beam_idx) = exp(2*pi*1i*(K_vec_x*D(1)+K_vec_y*D(2)));
    end
    H(:,:,beam_idx) = H(:,:,beam_idx).*taylor_win_El.*taylor_win_Az; % apply taylor
end

% Beampattern
phi_rad = deg2rad((-90:0.1:90)');
[phi_mat,m_mat] = meshgrid(phi_rad,0:20-1);
delta = VtrigU_ants_location(2)-VtrigU_ants_location(1);
f = Freqs(1);
D_phi = exp(-1i*2*pi*f*delta/3e8*m_mat.*sin(phi_mat));
Az_Directivity_dB = 20*log10(abs(D_phi'* ( squeeze(H(1:20:400, 1, :)) ) ));
El_Directivity_dB = 20*log10(abs(D_phi'* ( squeeze(H(1:20, 1, :)) ) ));
% % theta filters
% figure; subplot(121); plot(rad2deg(phi_rad), Az_Directivity_dB); ylim([-90 20]); title('Az')
% % phi filters
% subplot(122); plot(rad2deg(phi_rad), El_Directivity_dB); ylim([-90 20]); title('El')


rangeAzMap = zeros(Nfft/2, numOfDigitalBeams); % Range-Azimuth Map

for beam_idx = 1:numOfDigitalBeams
    %calculate and plot the steering response
    BR_response = ifft(mean(H(:,:,beam_idx).*Smat),Nfft,2);
    rangeAzMap(:, beam_idx) = BR_response(1:Nfft/2);
end


% plot Range-Azimuth Map
figure()
Ts = 1/Nfft/(Freqs(2)-Freqs(1)+1e-16); %Avoid nan checks
time_vector = 0:Ts:Ts*(Nfft-1);
r = time_vector(1:Nfft/2)*3e8/2; dB_Range = 80;
rangeAzMap_db = mag2db(abs(rangeAzMap)/max(abs(rangeAzMap(:))));
[xx, yy, img, h] = imageShow(rangeAzMap_db,r/1000,theta_vec,[-dB_Range 0]); 
close(h);
imagesc(xx*1000, yy*1000, img, [-dB_Range 0])
%     imagesc(rangeAzMap_db, [-dB_Range 0])
xlabel('Azimuth [rad]'); ylabel('Range [m]'); colormap jet; title(['Elevation: ' num2str(rad2deg(phi_s)) num2str(i)])
toc