function  [alpha ,v,iter] = L1CD(H,G,cm,cp,eps,max_iter)
% Function:  dual  coordinate descent  for the ftsvm
% Input:
% H,G,cm,cp,eps,max_iter
%
% Output:
% alpha ,v,iter
%
% Author: Bin-Bin Gao (csgaobb@gmail.com)
% Created on 2014.10.10
% Last modified on 2015.07.16

if ( nargin>6||nargin<4) % check correct number of arguments
    help L1CD
else
    [~,columnH]=size(H);
    l=size(G,1);
    
    if (nargin<5)
        eps=0.001;
    end
    if (nargin<6)
        max_iter=200;
    end
    
    
    E=eye(columnH);
    E(columnH,columnH)=0;
    
    Q_bar=(H'*H+cm*E)\G';
    for  i=1:l
        Q(i)=G(i,:)*Q_bar(:,i);
    end
        
    X_new = 1:l;
    X_old = 1:l;
    
    alpha  = zeros(l,1); 
    alphaold = zeros(l,1);
    v = zeros(columnH,1); 
    
    PGmax_old = inf;       %M_bar
    PGmin_old = -inf;      %m_bar
    
    iter = 1;    
    while iter<max_iter
        %1 While
        PGmax_new = -inf;   %M
        PGmin_new = +inf;   %m
        R = length(X_old);
        X_old = X_old(randperm(R));
        %2 for
        for  j = 1:R
            i = X_old(j);
            pg = -G(i,:)*v-1;  
            PG = 0;               
            if alpha(i) == 0
                if pg>PGmax_old
                    X_new(X_new==i) = [];
                    continue;
                elseif  pg<0
                    PG = pg;
                end
            elseif alpha(i)==cp(i)
                if pg<PGmin_old
                    X_new(X_new==i) = [];
                    continue;
                elseif  pg>0
                    PG = pg;
                end
            else
                PG = pg;
            end
            PGmax_new = max(PGmax_new,PG);
            PGmin_new = min(PGmin_new,PG);
            if abs(PG)> 1.0e-12
                alphaold(i,1) = alpha(i);
                alpha(i,1) = min(max(alpha(i)-pg/Q(i),0.0),cp(i));
                v = v-Q_bar(:,i)*(alpha(i,1)-alphaold(i,1));
            end
        end
        %     M(iter+1)=PGmax_new;
        %     N(iter+1)=PGmin_new;
        
        X_old = X_new;
        iter = iter+1; 
        %3
        if  PGmax_new-PGmin_new<=eps
            if length(X_old)==l
                break;
            else
                X_old = 1:l;  X_new = 1:l;
                PGmax_old = inf;   PGmin_old = -inf;
            end
        end
        
        %4 
        if  PGmax_new<=0
            PGmax_old = inf;
        else
            PGmin_old = PGmax_new;
        end
        %5 
        if  PGmin_old>=0
            PGmin_old = -inf;
        else
            PGmin_old = PGmin_new;
        end
        
    end;
    fprintf('convergent iteration times     : %d\n',iter);
end
end
