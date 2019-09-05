function V = signedVolume(x, t)

if nargin==1
    %% assume x contains tets which all have v1 at origin 0
    V = det3( [x(1:3:end, :) x(2:3:end, :) x(3:3:end, :)] );
else
    if size(t,2)~=4
        disp('not tetrahedron mesh');
        return;
    end

    % nf =(t,1);
    % V = zeros(nf, 1);
    % for i=1:nf
    % %     V(i) = det( X(T(i,2:4), :) - [1; 1; 1]*X(T(i,1), :) );
    %     V(i) = det( [x(t(i,:),:) ones(4,1)] );
    % end

    V = det3( [x(t(:,2), :)-x(t(:,1), :) x(t(:,3), :)-x(t(:,1), :) x(t(:,4), :)-x(t(:,1), :)] );
    % V = det3( reshape( x(t(:,2:4), :)', 9, nf )' - repmat(x(t(:,1), :),1,3) );
end

% if sum(V) < 0
%     V = -V;
% end

function d = det3(m)
% d = m(1,1)*(m(2,2)*m(3,3)-m(2,3)*m(3,2))...
% - m(1,2)*(m(2,1)*m(3,3)-m(2,3)*m(3,1))...
% + m(1,3)*(m(2,1)*m(3,2)-m(2,2)*m(3,1));

d = m(:,1).*(m(:,5).*m(:,9)-m(:,6).*m(:,8))...
- m(:,2).*(m(:,4).*m(:,9)-m(:,6).*m(:,7))...
+ m(:,3).*(m(:,4).*m(:,8)-m(:,5).*m(:,7));