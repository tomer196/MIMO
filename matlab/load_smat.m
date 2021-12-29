load('52.mat');
Smat = Smat1;

%Get antenna pairs and convert to matlab matrix
load('ants_location.mat');
ants_pairs = VtrigU_ants_location;
TxRxPairs = zeros(ants_pairs.Length,2);
for ii = 1: ants_pairs.Length
    TxRxPairs(ii,1) = double(ants_pairs(ii).tx);
    TxRxPairs(ii,2) = double(ants_pairs(ii).rx);
end

%Get used frequencies in Hz
Freqs = double(vtrigU.vtrigU.GetFreqVector_MHz())*1e6;

%%
%convert to complex time domain signal
Nfft = 2^(ceil(log2(size(Freqs,2)))+1);
Smat_td = ifft(Smat,Nfft,2);
Ts = 1/Nfft/(Freqs(2)-Freqs(1)+1e-16); %Avoid nan checks
time_vector = 0:Ts:Ts*(Nfft-1);

%Create and show power delay profile - non coherent summation
PDP = mean(abs(Smat_td),1);
figure; plot(time_vector*1.5e8,20*log10(abs(PDP./max(abs(PDP)))));ylim([-70 0]);
xlabel('Distance[m]');ylabel('Normalized amplitude[dB]');


%get antennas locations from script
vtrigU_ants_location;

%Create a steering vector
theta_vec = deg2rad(linspace(-60, 60, 128)); % Azimuth
phi = deg2rad(0); % Elevation

numOfDigitalBeams = numel(theta_vec); % number of digital rx beams
rangeAzMap = zeros(Nfft, numOfDigitalBeams); % Range-Azimuth Map
for beam_idx = 1:numOfDigitalBeams
    
    theta = theta_vec(beam_idx); % create new digital beam
    
    K_vec_x = 2*pi*Freqs*sin(theta)/3e8;
    K_vec_y = 2*pi*Freqs*sin(phi)/3e8;

    %Create a steering matrix for all pairs location
    H = zeros(size(TxRxPairs,1),size(Freqs,2));
    for ii = 1: size(TxRxPairs,1)
        D = VtrigU_ants_location(TxRxPairs(ii,1),:)+VtrigU_ants_location(TxRxPairs(ii,2),:);
        H(ii,:) = exp(2*pi*1i*(K_vec_x*D(1)+K_vec_y*D(2)));
    end


    %calculate and plot the steering response
    BR_response = ifft(mean(H.*Smat),Nfft,2);
    rangeAzMap(:, beam_idx) = BR_response;
    %hold on;plot(time_vector*1.5e8,20*log10(abs(BR_response./max(abs(BR_response)))));legend('Normalized non - Coherent summation','Normalized Coherent summation');
end
%% plot Range-Azimuth Map

r = time_vector*1.5e8; dB_Range = 50;
rangeAzMap_db = mag2db(abs(rangeAzMap)/max(abs(rangeAzMap(:))));
imageShow(rangeAzMap_db,r/1000,theta_vec,[-dB_Range 0]); 
xlabel('Azimuth [rad]'); ylabel('Range [m]'); colormap jet
title(['Elevation: ' num2str(rad2deg(phi))])