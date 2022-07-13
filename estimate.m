function [estGains, estDopplers, estDelays, estPhaseOffsets] = estimate(Hdd, alpha, beta, noiseVar, DividingNumber, M, N, Ncp, SamplingRate)
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