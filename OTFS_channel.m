% Copyright (c) 2021, KDDI Research, Inc. and KDDI Corp. All rights reserved.

addpath('functions');
addpath('classes');

clear variables
rng('default');
% rng('shuffle');
warning ('on','all');  % Turn on when testing

%% %%%%%%%%%%%%%%%%%%%%%%% INPUT PARAMETERS %%%%%%%%%%%%%%%%%%%%%%% %%

% System Parameters
nSubcarriersPerBlock = 12;  % Number of SCs / PRB
nSymbolsPerBlock = 14;  % Number of symbols per frame
fSubcarrierSpacing = 15e3;  % SC spacing in Hz
nFft = 256;  % FFT size
nCyclicPrefix = floor(nFft * 0.067);  % CP length in samples (6.7% is the calibration assumption defined in 3GPP R1-165989)
samplingRate = nFft * fSubcarrierSpacing;  % Sampling rate

modOrder = 16; % 4 for QPSK, 16 for 16QAM, 64 for 64QAM, 256 for 256QAM
nTxAntennas = 1;
nRxAntennas = 1;

nSubcarriers = nFft;
nBitsPerBlock = nSymbolsPerBlock * nSubcarriers * log2(modOrder);

% Parameter for LTE Channel Coder
turboDecodingNumIterations = 8;
codeRate = 0.5;  % Coding rate of source data bits over transmition redundant data bits
codedTransportBlockSize = nBitsPerBlock * codeRate;
crcBitLength = 24;  % defined in 3GPP TS36.2xx
maxCodeBlockSize = 6144;  % defined in 3GPP TS36.2xx
numCodeBlockSegments = ceil((codedTransportBlockSize - crcBitLength) / maxCodeBlockSize);
if numCodeBlockSegments > 1
    tbs = codedTransportBlockSize-crcBitLength*(numCodeBlockSegments+1);  % CRC24A + CRC24B*numSegments
else
    tbs = codedTransportBlockSize-crcBitLength;  % CRC24A only
end

% Parameters for Scrambler
rnti = hex2dec('003D');
cellId = 0;
frameNo  = 0; % radio frame
codeword = 0;

% Parameter for OFDM Modulator
nGuardBands = [(nFft - nSubcarriers)/2; (nFft - nSubcarriers)/2];  % Number of guard band subcarriers in both sides

% Parameter for ChannelEstimator
activeSubcarrierIndices = 1+nGuardBands(1):nFft-nGuardBands(2);

% Parameters for Fading Channel
velocity_kmph = 500;                   % user velocity [km/h]
delaySpread_ns = 300;                  % rms delay spred for TDL channel model only, in nano second
fCenter = 0.8e9;                       % center carrier frequency [Hz]
velocity = velocity_kmph*1000/(60*60); % convert km/h -> m/s
waveLength = physconst('lightspeed')/fCenter;
fDoppler = velocity / waveLength;
txCorrMat = 1;
rxCorrMat = 1;


%% %%%%%%%%%%%%%%%%%%%%%%% ALGORITHM SELECTION %%%%%%%%%%%%%%%%%%%%%%% %%

% ----- Path Estimator -----
channelEstimationAlgorithm = 'Pilot-based iterative path estimation';
% channelEstimationAlgorithm = 'PN-based iterative estimation';
% channelEstimationAlgorithm = 'Ideal';
fprintf("Channel estimation: %s\n", channelEstimationAlgorithm);

% ----- Equalizer -----
equalizerAlgorithm = 'Vectorized equalizer';
eqAlgorithm = 'blockedMMSE';
% equalizerAlgorithm = 'Deconvolutional equalizer';
% eqAlgorithm = 'Wiener';
fprintf("Channel equalization: %s\n", equalizerAlgorithm);
    
% ----- Channel -----
channelModel = 'EPA';
fprintf("Channel profile: %s\n", channelModel);

%---- Channel Coding ----
channelCoding = 'LTE';
% channelCoding = 'None';
fprintf("Channel coding: %s\n", channelCoding);

%% %%%%%%%%%%%%%%%%%%%%%%% OBJECT GENERATION %%%%%%%%%%%%%%%%%%%%%%% %%

switch channelCoding
    case 'None'
        % Bit Generator
        hBitGenerator = RandomBitGenerator('DataLength', nBitsPerBlock);
        % Symbol Mapper
        hSymbolMapper = QamMapperV2( ...
            'ModOrder', modOrder, ...
            'InputType', 'bit', ...
            'OutputType', 'bit', ...  % 'bit' for uncoded, 'llr' for coded
            'UnitAveragePower', true, ...
            'LLROverflowPrevention', true);
    case 'LTE'
        % Bit Generator
        hBitGenerator = RandomBitGenerator('DataLength', tbs);
        % Symbol Mapper
        hSymbolMapper = QamMapperV2( ...
            'ModOrder', modOrder, ...
            'InputType', 'bit', ...
            'OutputType', 'llr', ...  % 'bit' for uncoded, 'llr' for coded
            'UnitAveragePower', true, ...
            'LLROverflowPrevention', true);
        % Symbol Mapper for uncoded
        hSymbolMapperUncoded = QamMapperV2( ...
            'ModOrder', modOrder, ...
            'InputType', 'bit', ...
            'OutputType', 'bit', ...  % 'bit' for uncoded, 'llr' for coded
            'UnitAveragePower', true, ...
            'LLROverflowPrevention', true);
        % LTE Channel Coder
        hChannelCoder = LteChannelCoder( ...
            'TurboDecodingNumIterations', turboDecodingNumIterations, ...
            'ModOrder', modOrder, ...
            'NumLayers', 1, ...
            'OutputLength', nBitsPerBlock, ...
            'LinkDirection', 'Downlink', ...
            'RedundancyVersion', 0, ...
            'TBS', tbs);

        % Scrambler
        hScrambler = LteScrambler(rnti, cellId, frameNo, codeword);
end

% OFDM Modulator
hOfdmModulator = CpOfdmModulator(...
    'FFTLength',            nFft, ...
    'NumGuardBandCarriers', nGuardBands, ...
    'NumSymbols',           nSymbolsPerBlock, ...
    'CyclicPrefixLength',   nCyclicPrefix, ...
    'InsertDCNull',         false, ...
    'PilotInputPort',       false, ...
    'PilotOutputPort',      false, ...
    'NumTransmitAntennas',  nTxAntennas, ...
    'NumReceiveAntennas',   nRxAntennas, ...
    'SubcarrierIndexOrder', 'ZeroToFFTSize');

% OTFS Precoder
hPrecoder = OtfsPrecoder( ...
    'NumDelayBins', nSubcarriers, ...
    'NumDopplerBins', nSymbolsPerBlock, ...
    'NumTransmitAntennas', nTxAntennas, ...
    'NumReceiveAntennas', nRxAntennas);

% Path Estimator
switch channelEstimationAlgorithm
    case 'Pilot-based iterative path estimation'
        hEstimator = OtfsPilotResponseBasedPathParameterEstimator( ...
            'DividingNumber', 10, ...
            'CyclicPrefixLength', nCyclicPrefix, ...  % Ncp
            'NumDopplerBins', nSymbolsPerBlock, ...   % N
            'NumDelayBins', nSubcarriers, ...         % M
            'SamplingRate', samplingRate ...
            );
        threshAlpha = 1/50;
        threshBeta = 1/10;
        localUpdateThreshold = 0.01;
    case 'PN-based iterative estimation'
%         threshold = 1/150; % This is the best performance regardless the computation time
        threshold = 1/25;  % This is the same number of paths to be estimated with the proposed CE
        hEstimator = OtfsPNSeqBasedPathParameterEstimator( ...
            'DividingNumber', 10, ...
            'CyclicPrefixLength', nCyclicPrefix, ...  % Ncp
            'NumDopplerBins', nSymbolsPerBlock, ...   % N
            'NumDelayBins', nSubcarriers, ...         % M
            'SamplingRate', samplingRate, ...
            'Threshold', threshold);
    case 'Ideal'
end

% OTFS Equalizer
switch equalizerAlgorithm
    case 'Deconvolutional equalizer'
        hEqualizer = OtfsDeconvolutionalEqualizer( ...
            'NumSymbols', nSymbolsPerBlock, ...       % N
            'CyclicPrefixLength', nCyclicPrefix, ...  % Ncp
            'NumSubcarriers', nSubcarriers, ...       % M
            'SamplingRate', samplingRate, ...
            'DividingNumber', 10, ... 
            'EqualizationAlgorithm', eqAlgorithm);   
    case 'Vectorized equalizer'
        hEqualizer = OtfsVectorizedEqualizer( ...
            'EqualizationAlgorithm', 'ZF', ...
            'NumSymbols', nSymbolsPerBlock, ...       % N
            'CyclicPrefixLength', nCyclicPrefix, ...  % Ncp
            'NumSubcarriers', nSubcarriers, ...       % M
            'SamplingRate', samplingRate, ...
            'OutputConditionNumber', false, ...
            'EqualizationAlgorithm', eqAlgorithm);   
end

% Fading Channel
hFading = SoSBasedChannel( ...  % based on in-house implementation
    'ChannelModel', channelModel, ...
    'RMSDelaySpread', delaySpread_ns, ...
    'SamplingRate', samplingRate, ...
    'DopplerFrequency', fDoppler, ...
    'NumTxAntennas', nTxAntennas, ...
    'NumRxAntennas', nRxAntennas, ...
    'TxCorrMatrix', txCorrMat, ...
    'RxCorrMatrix', rxCorrMat, ...
    'OutputCIR', true, ...
    'CyclicPrefix', nCyclicPrefix, ...
    'FFTSize', nFft, ...
    'ImpulseSource', 'Input', ...
    'SequentialOrRandomChannel', 'Sequential', ...
    'FDFMethod', 'ApplyFDFToPathParameters');

% AWGN
hAwgn = AwgnChannel('N0');


%% %%%%%%%%%%%%%%%%%%%%%%% SIMULATION SETTINGS %%%%%%%%%%%%%%%%%%%%%%% %%
% Create an empty storage to store simulation status
snrRange = 10; % in dB:2:30
simIterations = 50 * ones(1,length(snrRange));

simstatus = table;  % use table-type variable since it's good to see in the workspace
simstatus.SNR = snrRange';
simstatus.BER = nan * zeros(length(snrRange),1);
simstatus.UncodedBER = nan * zeros(length(snrRange),1);
simstatus.BLER = nan * zeros(length(snrRange),1);
simstatus.TotalBitErrors = zeros(length(snrRange),1); 
simstatus.TotalUncodedBitErrors = zeros(length(snrRange),1); 
simstatus.TotalBlockErrors = zeros(length(snrRange),1); 
simstatus.Iteration = zeros(length(snrRange),1);
simstatus.PathCounts = zeros(length(snrRange),1);
simstatus.ComputeTime = zeros(length(snrRange),1);

time = strrep(strrep(strrep(string(datetime),'/',''),' ',''),':','');
filename = strcat('Resume-', mfilename, '-', time);

%% Load data when finding files that are created when simulation was interrupted
% 加载查找模拟中断时创建的文件的数据
resumefiles = ls(sprintf('Resume-%s-*.mat', mfilename));
if not(isempty(resumefiles))
    fprintf("Found file(s) to resume a simulation:\n")
    disp(resumefiles)
    for file = resumefiles'
        prompt = strcat('Do you want to resume the simulation using "', file', '"? [Y/n/F]:');
        answer = input(prompt, 's');
        if answer == 'Y'
            disp("Now Loading...")
            load(file, 'simstatus')  % Load only the simulation status
            snrRange = simstatus.SNR';
            filename = file;
            break;
        elseif answer == 'F'
            disp("Now Loading...")
            load(file)  % Load all parameters
            snrRange = simstatus.SNR';
            filename = file;
            break;
        end
    end
end


%% %%%%%%%%%%%%%%%%%%%%%%% START SIMULATION %%%%%%%%%%%%%%%%%%%%%%% %%
% For ideal channel estimation
delayDopplerImpulse = zeros(nSubcarriers, nSymbolsPerBlock); 
delayDopplerImpulse(1,1)=sqrt(nSymbolsPerBlock*nSubcarriers);
ddImpulseInTFDomain = hPrecoder.encode(delayDopplerImpulse);
ddImpulseInTimeDomain = hOfdmModulator.modulate(ddImpulseInTFDomain);

% For PN-sequence based channel estimation
switch channelEstimationAlgorithm
    case 'PN-based estimation'
        txPNSeq = zeros((nCyclicPrefix+nFft)*nSymbolsPerBlock*10,1);  % for long PN sequence
%         txPNSeq = zeros((nCyclicPrefix+nFft)*nSymbolsPerBlock*1,1);  % for short PN sequence
        nPNSeq = 1023;
        tmpSeq = repmat(genltegoldseq(nPNSeq, de2bi(31,31)), 1, ceil(length(txPNSeq)/nPNSeq));  % 1023-length with an initial value of 31
        tmpSeq(tmpSeq==0) = -1;  % map 0 or 1 bit into -1 or 1 bit
        txPNSeq = tmpSeq(1:length(txPNSeq))';
    case 'PN-based iterative estimation'
        txPNSeq = zeros((nCyclicPrefix+nFft)*nSymbolsPerBlock,1);
        nPNSeq = 1023;
        tmpSeq = repmat(genltegoldseq(nPNSeq, de2bi(31,31)), 1, ceil(length(txPNSeq)/nPNSeq));  % 1023-length with an initial value of 31
        tmpSeq(tmpSeq==0) = -1;  % map 0 or 1 bit into -1 or 1 bit
        txPNSeq = tmpSeq(1:length(txPNSeq))';
end

charCount = 49+17;
for snrdb = snrRange
    rng('default');
    rng(106);

    fprintf('\nSNR = %2d dB \n', snrdb);
    snrIndex = find(snrRange==snrdb);
    snr = 10^(snrdb/10);
    noiseVar = 1/snr;  % Total power
    N0 = noiseVar/nFft;  % N0 is the power spectral density of noise per unit of bandwidth, which is band-limited.

    totalBitErrors = 0;
    fprintf(repmat(' ', 1, charCount));  % Print spaces in advance to avoid deleting the previously displayed characters

    tic
    for count = simstatus.Iteration(snrIndex)+1:simIterations(snrIndex)
        %% Transmitter
        % Bit Generation
        txBits = hBitGenerator.generate();
%         txBits = zeros(size(txBits));  % FOR TEST
        switch channelCoding
            case 'None'
                txScrampledBits = txBits;
            case 'LTE'
                % Channel Encoding
                txCodedBits = hChannelCoder.encode(txBits);
                % Scrambler
                txScrampledBits = hScrambler.scramble(txCodedBits);
        end
        % Symbol Mapping (QAM modulation)
        txSymbols = hSymbolMapper.map(txScrampledBits);

        % OTFS Modulation
        txBlocks = reshape(txSymbols, nSubcarriers, nSymbolsPerBlock);
        txPrecodedBlocks = hPrecoder.encode(txBlocks);
        % OFDM Modulation
        txSignals = hOfdmModulator.modulate(txPrecodedBlocks);
        
        %% Channel

        % Fading Channel
        switch channelEstimationAlgorithm
            case {'Pilot-based iterative path estimation', 'Ideal'} 
                hFading.initRayleighFading();
                [distortedSignals, cir] = hFading.apply(txSignals, ddImpulseInTimeDomain);
            case 'PN-based iterative estimation'
                [distortedSignals, distortedPNSeq] = hFading.apply(txSignals, txPNSeq);
        end

        % AWGN Channel
        [noisySignals, noise] = hAwgn.add(distortedSignals, N0);
        
        %% Receiver
        % OFDM Demodulator
        rxPrecodedBlocks = hOfdmModulator.demodulate(noisySignals(1:(nFft+nCyclicPrefix)*nSymbolsPerBlock));
        % OTFS Demodulator
        rxBlocks = hPrecoder.decode(rxPrecodedBlocks);
        % Equalization (Ideal)
        switch channelEstimationAlgorithm
            case 'Pilot-based iterative path estimation'
                noisyCir = hAwgn.add(cir, N0);
                chanEst = hPrecoder.decode(hOfdmModulator.demodulate(noisyCir(1:(nFft+nCyclicPrefix)*nSymbolsPerBlock)));
                [estGains, estDopplers, estDelays, estOffsets] = estimate(chanEst, threshAlpha, threshBeta, noiseVar, 10, nSubcarriers, nSymbolsPerBlock, nCyclicPrefix, samplingRate);
                %[estGains, estDopplers, estDelays, estOffsets] = hEstimator.estimate(chanEst, threshAlpha, threshBeta, noiseVar);
                [Hi] = equalize(estGains, estDopplers, estDelays, estOffsets, nSubcarriers, nSymbolsPerBlock, nCyclicPrefix, samplingRate);
                Hi
                rxEqBlocks = hEqualizer.equalize(rxBlocks, estGains, estDopplers, estDelays, estOffsets, noiseVar, localUpdateThreshold);
            case 'PN-based iterative estimation'
                noisyPNSeq = hAwgn.add(distortedPNSeq, noiseVar*sqrt(nSubcarriers));
                [estGains, estDopplers, estDelays, estOffsets] = hEstimator.estimateIteratively(noisyPNSeq, txPNSeq, fDoppler*2);
                rxEqBlocks = hEqualizer.equalize(rxBlocks, estGains, estDopplers, estDelays, estOffsets, noiseVar, 0.01);
            case 'Ideal'
                chanEst = hPrecoder.decode(hOfdmModulator.demodulate(cir(1:(nFft+nCyclicPrefix)*nSymbolsPerBlock)));
                idealGains = hFading.pathRayleighGains;
                idealDopplers = hFading.pathDopplers;
                idealDelays = hFading.pathDelays;
                idealOffsets = hFading.pathOffsets;
                rxEqBlocks = hEqualizer.equalize(rxBlocks, idealGains, idealDopplers, idealDelays, idealOffsets, noiseVar);
        end
%         figure(1); clf; plot(rxEqBlocks, '.'); grid on; grid minor; xlim([-1.5 1.5]); ylim([-1.5 1.5]); % hold on; plot(rxIdealEqBlocks, '.'); hold off;
        rxSymbols = rxEqBlocks(:);
        % Symbol Demapping
        rxSoftBits = hSymbolMapper.demap(rxSymbols, noiseVar);
        % Channel Decoding
        switch channelCoding
            case 'None'
                rxUncodedHardBits = rxSoftBits;
                rxHardBits = rxUncodedHardBits;
                if sum(xor(txBits, rxHardBits)) > 0
                    blockError = 1;
                else
                    blockError = 0;
                end
            case 'LTE'
                rxUncodedHardBits = hSymbolMapperUncoded.demap(rxSymbols);
                % Descrambling
                rxDescrambledBits = hScrambler.descramble(rxSoftBits);
                [rxHardBits, blockError] = hChannelCoder.decode(rxDescrambledBits);
        end
        
        %% Result
        % Bit Error Rate
        numBitErrors = sum(xor(txBits, rxHardBits));
        simstatus.TotalBitErrors(snrIndex) = simstatus.TotalBitErrors(snrIndex) + numBitErrors;
        simstatus.BER(snrIndex) = simstatus.TotalBitErrors(snrIndex)/(count*nBitsPerBlock);
        simstatus.TotalBlockErrors(snrIndex) = simstatus.TotalBlockErrors(snrIndex) + blockError;
        simstatus.BLER(snrIndex) = simstatus.TotalBlockErrors(snrIndex)/count;
        numBitErrors = sum(xor(txScrampledBits, rxUncodedHardBits));
        simstatus.TotalUncodedBitErrors(snrIndex) = simstatus.TotalUncodedBitErrors(snrIndex) + numBitErrors;
        simstatus.UncodedBER(snrIndex) = simstatus.TotalUncodedBitErrors(snrIndex)/(count*length(txScrampledBits));
        fprintf(repmat('\b',1, charCount));  % Delete the prevously displayed characters, and then update the display
        fprintf('[%5d /%5d] BLER: %1.4f, BER: %1.7f (%9d/ %9d)', count, simIterations(snrIndex), simstatus.BLER(snrIndex), simstatus.UncodedBER(snrIndex), simstatus.TotalUncodedBitErrors(snrIndex), count*length(txScrampledBits));
        
        if mod(count,100)==0
            simstatus.Iteration(snrIndex) = count;
            save(filename);  % save all parameters to resume
        end
        pause(0.01)
    end
    computationTime = toc;
    computationTimePerIteration = computationTime/simIterations(snrIndex);
    fprintf('  Computation Time : %f sec. (as per iteration)\n', computationTimePerIteration);
    simstatus.ComputeTime(snrIndex) = computationTimePerIteration;
end

%% Save results
save(filename)

%% Show results
if usejava('jvm')  % if GUI is available
    figure
    semilogy(simstatus.SNR, simstatus.BER);
    ylabel('BER')
    xlabel('SNR (dB)')
    
    figure
    semilogy(simstatus.SNR, simstatus.BLER);
    ylabel('BLER')
    xlabel('SNR (dB)')     
end

function [estGains, estDopplers, estDelays, estPhaseOffsets, Hi] = estimate(Hdd, alpha, beta, noiseVar, DividingNumber, M, N, Ncp, SamplingRate)
    % ----- DESCRIPTION -----
    % This function estimates path parameters from the pilot response in the del y ay Doppler domain based on [1].
    % ----- INPUT PARAMETERS -----
    %   - Hdd: pilot response in the delay-Doppler domain (size: M x N)
    %   - alpha: tuning parameter for the summation of magnitude
    %   - beta: tuning parameter for noise variance
    %   - noiseVar: noise variance
    % ----- REFERENCE -----
    % [1] https://arxiv.org/abs/2010.15396      
            
    % - Hdd：延迟多普勒域中的导频响应（大小：M x N）
    % - alpha：幅度总和的调整参数
    % - beta：噪声方差的调整参数
    % -noiseVar：噪声方差 
    
    kappa = 0:1/DividingNumber:1-1/DividingNumber;
            
    upsilon = zeros(N,length(kappa));
    for k = 1:length(kappa)
        x = kappa(k) - (0:N-1);
        for i = 1:N
            upsilon(:,k) = upsilon(:,k) + 1/N * exp(1i*2*pi*(i-1)*x/N).';
        end
    end
    
    estGains = [];
    estDopplers = [];
    estDelays = [];
    estPhaseOffsets = [];
    Hdd = Hdd/sqrt(M*N);

    kappa = 0:1/DividingNumber:1-1/DividingNumber;
    pathindex = 1;
    for delay = 0:M-1
        Hprime = Hdd(delay+1,:);
        % 此延迟域中的振幅总和
        sumMagnitude = abs(sum(Hdd(delay+1,:)));  % Sum of amplitudes in this delay bin
        estGain = Inf;
        while(1)
            % Take cross-correlation b/w Hdd and Upsilon
            % 获得 Hdd 和 Upsilon 的互相关
            crosscorr = zeros(length(kappa), N);
            for idx_kappa = 1:length(kappa)
                crosscorr(idx_kappa,:) = ifft( fft(Hprime) .* conj(fft(upsilon(:,idx_kappa).')) );
            end
            crosscorrvec = circshift(crosscorr(:), N*DividingNumber/2);

            % Condition 1: find paths until the cross-correlation becomes small
            % 条件1：寻找路径直到互相关变小
            [corrval, largestindices] = sort(abs(crosscorrvec), 'descend');
            % 删除小于阈值的索引
            largestindices(corrval < alpha * sumMagnitude) = [];  % remove indices that are less then the threshold

            % Condition 2: the cross-correlation is not negligibly small (equally or less than the noise power)
            % 条件2：互相关小，但不可忽略（等于或小于噪声功率）
            % 删除小于噪声功率的索引
            largestindices(abs(crosscorrvec(largestindices)) < beta * sqrt(noiseVar)) = [];  % remove indices that are less then the noise power

            if isempty(largestindices)
                break;
            end

            largestidx = largestindices(1);

            if estGain < abs(crosscorrvec(largestidx))
                break;
            end
            estGain = abs(crosscorrvec(largestidx));
            estDoppler = (largestidx-1)/DividingNumber-N/2;  % index -> value of Doppler
            estDopplerShift = exp(1i*2*pi*estDoppler*(Ncp-delay)/((M+Ncp)*N));
            estInitialPhase = crosscorrvec(largestidx)/abs(crosscorrvec(largestidx))*estDopplerShift^-1;

            estGains = [estGains; estGain];
            estDopplers = [estDopplers; estDoppler];
            estDelays = [estDelays; delay];
            estPhaseOffsets = [estPhaseOffsets; angle(estInitialPhase)];

           % Check estimated paths
%            figure; clf;
%            plot((0:1/DividingNumber:N-1/DividingNumber)-N/2, abs(crosscorrvec)); hold on; 
%            stem(estDopplers(pathindex), estGains(pathindex));
%            xlim([-N/2 N/2]); ylim([0 1]);
%            xlabel('k+\kappa'); ylabel('|R_H_,_\Upsilon(k+\kappa)|');

            intDoppler = floor(estDopplers(pathindex));
            fracDopplerIdx = floor(mod(estDopplers(pathindex),1)*DividingNumber + 1 + 1e-9);
            Hprime = Hprime - estGains(pathindex) * exp(1i*estPhaseOffsets(pathindex))*circshift(upsilon(:,fracDopplerIdx),intDoppler).';

            pathindex = pathindex + 1;
        end
    end
            
    estDopplers = estDopplers / N / ((M+Ncp)/SamplingRate);
    estDelays = estDelays / SamplingRate;
%   estPhaseOffsets = estPhaseOffsets - angle(exp(1i*2*pi*estDopplers*(M+Ncp)*N/this.SamplingRate));

    numPaths = length(estGains);
    estDelaysInSample = estDelays*SamplingRate;
    estDopplersInSample = estDopplers/(SamplingRate/((M+Ncp)*N));
    
    
%     HddEst = zeros(M, N);
%     for p = 1:numPaths
%         for l = 0:M-1
%             if estDelaysInSample(p) == l
%                 Upsilon_N = zeros(1,N);
%                 for i = 1:N
%                     x = (estDopplers(p)/(SamplingRate/((M+Ncp)*N))-[0:N-1]);
%                     Upsilon_N = Upsilon_N + 1/N * exp(1i*2*pi*(i-1)*x/N);
%                 end                        
%                 HddEst(l+1,:) = HddEst(l+1,:) + estGains(p) * exp(1i*estPhaseOffsets(p)) * exp(1i*2*pi*estDopplersInSample(p)*(Ncp-estDelaysInSample(p)+l)/((M+Ncp)*N)) * Upsilon_N;
%             end
%         end
%     end
end

function [Hi] = equalize(pathGains, pathDopplers, pathDelays, pathOffsets, M, N, Ncp, SamplingRate)
    % ----- DESCRIPTION -----
    % This function re-generates the channel matrix using path parameters.
    % Then, the received symbols are equalized by multiplying the inverse matrix of the generated channel matrix.
    % ----- INPUT PARAMETERS -----9999999999999999999999999999999
    %   - Ydd: received symbols in the Delay doppler domain
    %   - pathGains, pathDopplers, pathDelays, pathOffsets: path parameters
    %   - noiseVar: noise variance for MMSE
            
    % Convert fractional delays into integer delays
    [fdfedPathGains, fdfedPathDopplers, fdfedPathDelays, fdfedPathOffsets] = fdfpathpars(pathGains, pathDopplers, pathDelays, pathOffsets, SamplingRate);
            
    % Initialize parameters to be short names
    P = length(fdfedPathGains);

    % Delay-Doppler domain grid representation
    fdfedPathDelaysInSample = fdfedPathDelays*SamplingRate;
    fdfedPathDopplersInSample = fdfedPathDopplers * N * ((M+Ncp)/SamplingRate);

    %% Re-generate OFDM-symbol-wise (i: symbol number) time domain channel matrix: Hi (size: M x M)
    % Initialize matrixes
    Hi = cell(N,1);
    for i = 1:N
        Hi{i} = zeros(M,M);
    end

    % Re-generate Hi
    for p = 1:P
        for i = 1:N
            timeIndices = (((M+Ncp)*(i-1)+Ncp) : ((M+Ncp)*i-1)) - fdfedPathDelaysInSample(p);
            phaseShift = exp(1i * (2*pi * fdfedPathDopplersInSample(p) / ((M+Ncp)*N) * timeIndices)); 
            dopplerMat = diag(phaseShift);
            Hi{i} = Hi{i} + fdfedPathGains(p) * exp(1i*fdfedPathOffsets(p)) * circshift(dopplerMat, -fdfedPathDelaysInSample(p), 2); 
        end
    end
end

% function [fdfedPathGains, fdfedPathDopplers, fdfedPathDelays, fdfedPathOffsets] = fdfpathpars(pathGains, pathDopplers, pathFractionalDelays, pathOffsets, samplingRate)
%     % ### DESCRIPTION ###
%     % This function converts path parameters with fractional delays to path parameters with integer 
%     % delays using the fractional delay filter. 
%     % 
%     % ### INPUT PARAMETERS ###
%     %   - pathGains: path gains
%     %   - pathDopplers: path Doppler shifts 
%     %   - pathFractionalDelays: path delays with a fractional delay
%     %   - pathOffsets: initial offset of each path
%     %   - varargin: if there is an another parameter
% 
%     delays=pathFractionalDelays*samplingRate;
%     fdfCoeffs = cell(length(delays),1);
%     nonZeroSampIndices = zeros(length(delays),1);
% 
% 
%             
%     for delayIndex = 1:length(delays)
%         delay = delays(delayIndex);
%         filterOrder = min(floor(2*delay+1), FilterOrder);  
%         % Note: The filter must be causal. The filter order is selected based on the delay to be causal and the parameter.
% 
%         % Calc a modified coefficient matrix of modified Farrow structure
%         % Note: This algorithm is based on a book below.
%         % [1] V鋖im鋕i, V., and T. I. Laakso. 揊ractional Delay Filters桪esign and Applications.? In Nonuniform Sampling, edited by Farokh Marvasti, 835?95. Information Technology: Transmission, Processing, and Storage. Boston, MA: Springer US, 2001. https://doi.org/10.1007/978-1-4615-1229-5_20.
%         N = filterOrder;
%         U = fliplr(vander(0:N));
%         Q = invcramer(U);  % solve inverse matrix using Cramer's rule
%         T = zeros(N+1,N+1);
%         for n = 0:N
%             for m = 0:N
%                 if n >= m
%                     T(m+1,n+1) = round(N/2)^(n-m) * nchoosek(n,m);
%                 end
%             end
%         end
%         Qf = T*Q;
%                 
%         % Calculate filter coefficients of the FIR filter
%         fracDelay = mod(delay,1);
%         delayVec = fracDelay.^(0:filterOrder);
%         this.fdfCoeffs{delayIndex} = delayVec * Qf;
% 
%         % Calculate non zero sample index (=start index)
%         if mod(filterOrder,2)==0  % even
%             this.nonZeroSampIndices(delayIndex) = round(delay+0.5) - filterOrder/2;  % non-zero sample index, written as equation (3.37) in [1]
%         else  % odd
%             this.nonZeroSampIndices(delayIndex) = floor(delay) - (filterOrder-1)/2;
%         end
%     end
%             
%         % Normalize the filter coefficient when a high filter order (since the accuracy of the filter is degraded)
%     if FilterOrder > 9
%         sinewave = this.apply(sin(0:pi/40:4*pi));
%         calib = rms(sinewave(nonZeroSampIndices(end)+maxFilterOrder+20:nonZeroSampIndices(end)+maxFilterOrder+100-1,:))' / sqrt(1/2);
%         for delayIndex = 1:length(delays)
%             this.fdfCoeffs{delayIndex} = this.fdfCoeffs{delayIndex}/calib(delayIndex);
%         end
%     end
% 
%     totalNumPars = 0;
%     for p = 1:length(pathFractionalDelays)
%         totalNumPars = totalNumPars + length(fdfCoeffs{p}) - 1;
%     end
% 
%     fdfedPathGains = zeros(totalNumPars,1);
%     fdfedPathDopplers = zeros(totalNumPars,1);
%     fdfedPathDelays = zeros(totalNumPars,1);
%     fdfedPathOffsets = zeros(totalNumPars,1);
% 
%     newPathIdx = 1;
%     for p = 1:length(pathFractionalDelays)
%         for d = 1:length(fdfCoeffs{p})
%             if d - 2 + FDF.nonZeroSampIndices(p) < 0 % ignore non causal delay index
%                 continue;
%             end
%             fdfedPathGains(newPathIdx) = pathGains(p) * fdfCoeffs{p}(d);
%             fdfedPathDopplers(newPathIdx) = pathDopplers(p);
%             fdfedPathDelays(newPathIdx) = (d - 2 + FDF.nonZeroSampIndices(p))/samplingRate;
%             fdfedPathOffsets(newPathIdx) = angle( exp( 1i * ( 2*pi*pathDopplers(p)*fdfedPathDelays(newPathIdx) + pathOffsets(p)) ) );
%         
%             newPathIdx = newPathIdx + 1;
%         end
%     end
% 
%     % Filter very small path
%     thresh = 1e-5;
%     fdfedPathDopplers(abs(fdfedPathGains) < thresh) = [];
%     fdfedPathDelays(abs(fdfedPathGains) < thresh) = [];
%     fdfedPathOffsets(abs(fdfedPathGains) < thresh) = [];
%     fdfedPathGains(abs(fdfedPathGains) < thresh) = [];
% 
% end