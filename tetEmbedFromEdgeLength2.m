function y = tetEmbedFromEdgeLength2(el01, el02, el03, el12, el13, el23)

% find tet with given edge length squares, el2s: 12 13 14 23 24 34   % in c index 01 02 03 12 13 23
% see paille et al 2015, appendex C trilateration

x2 = (el23 + el13 - el03)/2./sqrt(el23);
y2 = sqrt( el13 - x2.^2 );
x3 = (el23 + el12 - el02)/2./sqrt(el23);
y3 = (el12 - el01 - 2*x2.*x3 + x2.^2 + y2.^2)/2./y2;
z3 = sqrt( el12 - x3.^2 - y3.^2 );

nt = numel(el01);
y = zeros(nt*3, 3);
y(1:3:end, :) = [sqrt(el23) zeros(nt, 2)];
y(2:3:end, 1:2) = [x2 y2]; 
y(3:3:end, :) = [x3 y3 z3]; 