function [Hi] = equaliz(fdfedPathGains, fdfedPathDopplers, fdfedPathDelays, fdfedPathOffsets, M, N, Ncp, SamplingRate)
    % ----- DESCRIPTION -----
    % This function re-generates the channel matrix using path parameters.
    % Then, the received symbols are equalized by multiplying the inverse matrix of the generated channel matrix.
    % ----- INPUT PARAMETERS -----9999999999999999999999999999999
    %   - Ydd: received symbols in the Delay doppler domain
    %   - pathGains, pathDopplers, pathDelays, pathOffsets: path parameters
    %   - noiseVar: noise variance for MMSE
            
    % Convert fractional delays into integer delays
   % [fdfedPathGains, fdfedPathDopplers, fdfedPathDelays, fdfedPathOffsets] = fdfpathpars(pathGains, pathDopplers, pathDelays, pathOffsets, SamplingRate);
     fdfedPathDelays=floor(fdfedPathDelays+0.5);    
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
