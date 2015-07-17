function [d,x]=distance(xi,X)
%  Author: Bin-Bin Gao (csgaobb@gmail.com)
% Created on 2014.10.10
% Last modified on 2015.07.16

if ( nargin>2||nargin<2) % check correct number of arguments
    help distance
else
    [rx,~]=size(X);
    for  i=1:rx
        di(i,1)=norm(xi-X(i,:));
    end
    d =min(di);
    x=X(di(:,1)==d,:);
    x=x(1,:);
end
end