function [distance_xiX,xixing]=distance(xi,X)
%  Author: Bin-BinGaa (csgaobb@gmail.com)
% Created on 2014.10.10
% Last modified on 2015.07.16


if ( nargin>2||nargin<2) % check correct number of arguments
    help distance
else
    [rx,cx]=size(X);
    for  i=1:rx
        distance_per(i,1)=norm(xi-X(i,:));
    end
    distance_xiX=min(distance_per);
    xxxx=X(distance_per(:,1)==distance_xiX,:);
    xixing=xxxx(1,:);
end
end