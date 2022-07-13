close all
clear all
rng(1)
%% OTFS parameters%%%%%%%%%%
% N: number of symbols in time
N = 8;
% M: number of subcarriers in frequency
M = 64;
% M_mod: size of QAM constellation
M_mod = 4;
M_bits = log2(M_mod);
% average energy per data symbol
eng_sqrt = (M_mod==2)+(M_mod~=2)*sqrt((M_mod-1)/6*(2^2));


%% delay-Doppler grid symbol placement
%%采用在信道传输过程中加cp的方式，不采用在消息调制时加zp的方式
%number of symbols per frame
 N_syms_perfram =M*N;
% number of bits per frame
N_bits_perfram = N_syms_perfram*M_bits;

% Time and frequency resources
car_fre = 4*10^9;% Carrier frequency
delta_f = 15*10^3; % subcarrier spacing: 15 KHz
T = 1/delta_f; %one time symbol duration in OTFS frame
Ncp=16;
M_cp=M+Ncp;

% SNR and variance of the noise
% SNR = P/\sigma^2; P: avg. power of albhabet transmitted
SNR_dB = 6:2:30;
SNR = 10.^(SNR_dB/10);
sigma_2 = (abs(eng_sqrt)^2)./SNR;


%% Initializing simulation error count variables
N_fram = 1000;%传输2次

%%Initialize_error_count_variables;
est_info_bits_LMMSE=zeros(N_bits_perfram,1);
err_ber_LMMSE = zeros(1,length(SNR_dB));
avg_ber_LMMSE=zeros(1,length(SNR_dB));

%% Normalized DFT matrix@
Fn=dftmtx(N);  % Generate the DFT matrix
Fn=Fn./norm(Fn);  % normalize the DFT matrix
current_frame_number=zeros(1,length(SNR_dB));
%% OTFS channel generation
%%channel model following 3GPP standard
max_speed=500;  % km/hr
[chan_coef,delay_taps,Doppler_taps,taps]=Generate_delay_Doppler_channel_parameters(N,M_cp,car_fre,delta_f,T,max_speed);
L_set=unique(delay_taps);
[G,gs]=Gen_time_domain_channel(N,M_cp,taps,delay_taps,Doppler_taps,chan_coef);

for iesn0 = 1:length(SNR_dB)%%不同信噪比下的情况，6和20的情况较好
    for ifram = 1:N_fram%%第一帧传导频，之后都传消息@
        current_frame_number(iesn0)=ifram;
        %% random input bits generation%%%%%
        if (ifram~=1)
        trans_info_bit = randi([0,1],N_syms_perfram*M_bits,1);
        %%2D QAM symbols generation %%%%%%%%
        data=qammod(reshape(trans_info_bit,M_bits,N_syms_perfram), M_mod,'gray','InputType','bit');
       % X = Generate_2D_data_grid(N,M,data,data_grid);%要发送的信息填入2D晶格
        X=reshape(data,M,N);
        else
        Hd=zeros(M,N); 
        Hd(1,1)=(M*N)^(1/2); 
        X=Hd;
        end
        
        %% OTFS modulation%%%%@
        X_tilda=X*Fn';
        
        X_tilda_cp=zeros(M_cp,N);
        X_tilda_cp(1:M,:)=X_tilda;
        X_tilda_cp(M+1:M_cp,:)=X_tilda(1:Ncp,:);
        s = reshape(X_tilda_cp,N*M_cp,1);
   
        %%%%%模拟实际信道传输
%         %% OTFS channel generation
%         % channel model following 3GPP standard
%         max_speed=500;  % km/hr
%         [chan_coef,delay_taps,Doppler_taps,taps]=Generate_delay_Doppler_channel_parameters(N,M_cp,car_fre,delta_f,T,max_speed);
%         L_set=unique(delay_taps);
        %% channel output
        %[G,gs]=Gen_time_domain_channel(N,M_cp,taps,delay_taps,Doppler_taps,chan_coef);
        rcp=zeros(N*M_cp,1); 
        noise= sqrt(sigma_2(iesn0)/2)*(randn(size(s)) + 1i*randn(size(s)));
        for q=1:N*M_cp
            for l=(L_set+1)
                if(q>=l)
                    rcp(q)=rcp(q)+gs(l,q)*s(q-l+1);
                end
            end
        end
        rcp=rcp+noise;
        rs=reshape(rcp,M_cp,N);
        r=rs(1:M,:);%remove cp
        r=reshape(r,M*N,1);
       

       %% OTFS demodulation%%%%导频测信道先经过解调@
       Hdd_tilda=reshape(r,M,N);
       Hdd = Hdd_tilda*Fn;

        %% channel estimate@
        if(ifram==1)
        %noiseVar=1./SNR(iesn0);
        noiseVar=sigma_2(iesn0);
        alpha = 1/50;
        beta = 1/10;
        DividingNumber=10;
        SamplingRate=delta_f*M;
        [estGains, estDopplers, estDelays, estPhaseOffsets] = estimate(Hdd, ...
            alpha, beta, noiseVar, DividingNumber, M, N, Ncp, SamplingRate);
        [Hi] = equaliz(estGains, estDopplers, estDelays, estPhaseOffsets, ...
            M, N, Ncp, SamplingRate);
        end

        %% LMMSE detector@
        [est_info_bits_LMMSE,data_LMMSE] =LMMSE(N,M,M_mod, ...
           sigma_2(iesn0),r,Hi);
       
        %% errors count%%%%%
        if(ifram~=1)
        count_errors_per_frame;
        end

        %% DISP error performance details
        display_errors_per_frame;
       
    end
    
end

figure(1)
semilogy(SNR_dB,avg_ber_LMMSE,'-s','LineWidth',2,'MarkerSize',8)
legend('block-wise time-domain LMMSE in [R3]')
grid on
xlabel('SNR(dB)')
ylabel('BER')