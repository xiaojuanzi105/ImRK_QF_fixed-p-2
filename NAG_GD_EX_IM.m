clear all;
clc;
close all;
% % %% real data set
 L=0*2*10^(-10);
N=100;
% % 
a=1;
% % %%
% % % load('colon_cancer_A');
% % % load('colon_cancer_label');
% % % W=A';
% % % H=Y;
% % % c_W =size(W,1);
% % % for i=1:c_W
% % % Ma= max(W(i,:));
% % % Mi = min(W(i,:));
% % %  W(i,:) = (W(i,:) - Mi)/(Ma-Mi);
% % %  H(i) = (H(i,:) - Mi)/(Ma-Mi);
% % % end
% % % load('X_a1a');
% % % load('y_a1a_label');
%% 只有2级的显式RK June 12
% h1=0.04;
% h2=0.08;
% h3=0.16;
% 
% h4=0.02;
% h_NAG=0.02;
% h_GD=0.02;

%%
K=1;

SW1=1;
SW2=2;

ImF1=zeros(N+1,K);
ImF2=zeros(N+1,K);
ImF3=zeros(N+1,K);

ExF1=zeros(N+1,K);
ExF2=zeros(N+1,K);
ExF3=zeros(N+1,K);

ExF_NAG=zeros(N+1,K);
ExF_GD=zeros(N+1,K);
%%
ImF11 = zeros(N+1,1);
ImF22 = zeros(N+1,1);
ImF33 =  zeros(N+1,1);
ExF11 =  zeros(N+1,1);
ExF22 =  zeros(N+1,1);
ExF33 = zeros(N+1,1);
ExF_NAG1 =  zeros(N+1,1);
ExF_GD1 =  zeros(N+1,1);
%%

% h1=0.02;
% h2=0.1;
% h3=0.2;
% h_NAG=0.02;
% h_GD=0.02;
%% 临界值
% % % h1=0.01;
% % % h2=0.05;
% % % h3=0.18;
% % % 
% % % h6=0.01;
% % % h4=0.05;
% % % h5=0.18;
%% fixed W H step size June 12 
h1=0.04;
h2=0.06;
h3=0.2;

h6=0.01;
h4=0.02;
h5=0.15;
h_NAG=0.01;
h_GD=0.01;


for k=1:K
%%
X=rand(10,10);
y=rand(10,1);
for i=1:10
    if y(i) >= 0.5
        y(i)=1;
    else
        y(i)=0;
    end
end
% % 
W=X;
H=y; 
d=size(W,2);
 %% initial conditions
z11=rand(d,1); z22=zeros(d,1); z33=1;
z0=[z11;z22;z33];
%%
ImX1=Im_1s1o_trape(a, h1, N, z0, d, W, H, L);
ImX2=gauss_bfgs_crj(a, h2, N, z0, d, W, H, L);
ImX3=Im_3s6o_gauss(a, h3, N, z0, d, W, H, L);
 ExX1=Ex_1s1o_Kutta(a, h6, N, z0, d, W, H, L);
ExX2=Ex_2s2o_Kutta(a, h4, N, z0, d, W, H, L);
 ExX3=Ex_3s3o_Heun(a, h5, N, z0, d, W, H, L);

X_NAG=nesterov(h_NAG, N, z0, d, W, H, L, SW2);
X_GD=nesterov(h_GD, N, z0, d, W, H, L, SW1);
for j=1:N+1
%% loss 
ImF1(j, k) = Fu(ImX1(:, j), W, H, L);
ImF2(j, k) = Fu(ImX2(:, j), W, H, L);
ImF3(j, k) = Fu(ImX3(:, j), W, H, L);

 ExF1(j, k) = Fu(ExX1(:, j), W, H, L);
ExF2(j, k) = Fu(ExX2(:, j), W, H, L);
 ExF3(j, k) = Fu(ExX3(:, j), W, H, L);

ExF_NAG(j, k) = Fu(X_NAG(:, j), W, H, L);
ExF_GD(j, k) = Fu(X_GD(:, j), W, H, L);
end
%%
ImF22 = ImF22 + ImF2(:, k);
ImF33 = ImF33 + ImF3(:, k);
ImF11 = ImF11 + ImF1(:, k);

ExF11 = ExF11 + ExF1(:, k);
ExF22 = ExF22 + ExF2(:, k);
ExF33 = ExF33 + ExF3(:, k);

ExF_NAG1 = ExF_NAG1 + ExF_NAG(:, k);
ExF_GD1 = ExF_GD1 + ExF_GD(:, k); 
end
figure
% semilogy(1:N+1, ImF1,'r-','LineWidth', 1.5);hold on
% semilogy(1:N+1, ImF2,'b-','LineWidth', 1.5);hold on
% semilogy(1:N+1, ImF3,'k-','LineWidth', 1.5);hold on
% 
% semilogy(2:N+1, ExF2(2:end),'bo--','LineWidth', 1.5);hold on
% semilogy(2:N+1, ExF3(2:end),'ro--','LineWidth', 1.5);hold on
% 
% % semilogy(1:N+1, ExFF,'b--','LineWidth', 1.5);
% semilogy(1:N+1, ExF_NAG,'k:','LineWidth', 2.5);hold on;
% semilogy(1:N+1, ExF_GD,'k-.','LineWidth', 1.5);


loglog(1:N+1, ImF11(1:end)/K, 'r-','LineWidth', 1.5);hold on
loglog(1:N+1, ImF22(1:end)/K,'b-','LineWidth', 1.5);hold on
loglog(1:N+1, ImF33(1:end)/K,'k-','LineWidth', 1.5);hold on

 loglog(1:N+1, ExF11(1:end)/K,'r--','LineWidth', 1.5);hold on
loglog(1:N+1, ExF22(1:end)/K,'b--','LineWidth', 1.5);hold on
 loglog(1:N+1, ExF33(1:end)/K,'g--','LineWidth', 1.5);hold on

loglog(1:N+1, ExF_NAG1(1:end)/K,'k:','LineWidth', 2.5);hold on;
loglog(1:N+1, ExF_GD1(1:end)/K,'k-.','LineWidth', 1.5);
xlabel('Iterations', 'FontSize',16);
ylabel('Objective','FontSize',16);
% ylim([10^(-14),1000]);
%title('Minimizing regularized quadratic function on  set','FontSize',16);
% legend({'ImRK s=1','ImRK s=2','ImRK s=3'...
%,'ExRK s=2','ExRK s=3','NAG','GD'},'FontSize',16); %'Ex-2s2o-Kutta'
legend({'ImRk s=1','ImRk s=2','ImRk s=3',...
    'ExRk s=1','ExRk s=2','ExRk s=3','NAG','GD'},...
    'FontSize',16,'Location', 'southwest');
% legend({'ImRk s=1','ImRk s=2','ImRk s=3',...
%    'ExRk s=2','NAG','GD'},...
%     'FontSize',16,'Location', 'southwest');
set(gca,'FontSize',16);