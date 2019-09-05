function x = alignToAnchors(x, anchors)

ancId = anchors(:,1);
ancPos = anchors(:,2:end);
nAnc = numel(ancId);

assert(size(anchors,2)==4 && nAnc==3);  % id + x,y,z coords

% use covariance instead of affine transformation, which is not well-defined for 2 vector in 3d
% aff = (x(ancId(2:end),:)-x(ancId(1),:)) \ (ancPos(2:end, :)-ancPos(1, :));
cov = (x(ancId(2:end),:)-x(ancId(1),:))'*(ancPos(2:end, :)-ancPos(1, :));

[u, s, v] = svd( cov );
if det(u*v')<0
    s(end, end) = s(end, end);
    v(:, end) = -v(:, end);
end
S = u*v';
% S = u*s*v';

%% apply global transformation with centering
x = (x - x(ancId(1),:))*S;
x = x + ancPos(1, :);