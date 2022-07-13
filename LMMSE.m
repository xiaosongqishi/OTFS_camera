function [est_bits,x_data] = LMMSE(N,M,M_mod,noise_var,r,Hi)
%% Normalized DFT matrix
Fn=dftmtx(N);  % Generate the DFT matrix
Fn=Fn./norm(Fn);  % normalize the DFT matrix
%% Initial assignments
%Number of symbols per frame
 N_syms_perfram=M*N;%如果cp为0，则符号数为MN

%number of bits per QAM symbol
M_bits=log2(M_mod);
%number of bits per frame
N_bits_perfram = N_syms_perfram*M_bits;
%received time domain blocks 
sn_block_est=zeros(M,N);

%% block-wise LMMSE detection
for n=1:N    
    rn=r((n-1)*M+1:n*M); %取每一列单独运算   
    Rn=Hi{n}'*Hi{n};
    sn_block_est(:,n)=(Rn+noise_var.*eye(M))^(-1)*(Hi{n}'*rn);
end
X_tilda_est=sn_block_est;
%% detector output
X_est=X_tilda_est*Fn;
x_est=reshape(X_est,1,N*M);
x_data=x_est;%cp为0，所有得符号都为传输符号
est_bits=reshape(qamdemod(x_data,M_mod,'gray','OutputType','bit'),N_bits_perfram,1);
end
