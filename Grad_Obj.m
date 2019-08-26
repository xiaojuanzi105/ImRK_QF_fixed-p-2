%% The objective function
function f = Grad_Obj(d, t, A, L)
%%
% f = (A'*A)*y - A'*b;
% d=size(y);
f11 =zeros(d);
f12 = eye(d);
f13 =zeros(d,1);

f21= -4*(A'*A)- 4*L*eye(d);
f22= -(5/t)*eye(d);
f23= -(5/t^2)*eye(d,1);

f31=zeros(1,d);
f32=zeros(1,d);
f33=0;
f=[f11, f12, f13; f21, f22, f23;f31, f32, f33];
end

% %% The objective function
% function f = Grad_Obj(d, t, A, L)
% %%
% % f = (A'*A)*y - A'*b;
% % d=size(y);
% f11 =zeros(d);
% f12 = eye(d);
% f13 =zeros(d,1);
% 
% tau = b .* (A * x);
% f21= -4*(A'*(b./((1  + exp(tau)))^2)* (A'*(b.*exp(tau))+L));
% f22= -(5/t)*eye(d);
% f23= -(5/t^2)*eye(d,1);
% 
% f31=zeros(1,d);
% f32=zeros(1,d);
% f33=0;
% f=[f11, f12, f13; f21, f22, f23;f31, f32, f33];
% end

