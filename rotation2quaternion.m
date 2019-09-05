function q = rotation2quaternion(M)

% https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
 

n = size(M, 1)/3;
q = zeros(n, 4);

ibase = (0:n-1)'*3;
diagM = [M(ibase+1, 1) M(ibase+2, 2) M(ibase+3, 3)];
t = sum(diagM, 2);

%%
i = t > 0;
if any(i) % need for empty matrix, s
    r = sqrt(1+t(i));
    s = 1/2./r;
    base = ibase(i);
    q(i,:) = [r/2 M(base+3,2)-M(base+2,3)  M(base+1,3)-M(base+3,1)  M(base+2,1)-M(base+1,2)];
    q(i,2:4) = q(i,2:4).*s;
end

%%
i = t<=0;
if any(i)
    base = ibase(i);
    [~, j1] = max(diagM(i,:), [], 2);
    r = sqrt( 1 - t(i) + 2*M( sub2ind(size(M), base+j1, j1) ) );
    s = 1/2./r;
    s = reshape(s, [], 1); % need for empty matrix

    j2 = mod(j1,3) + 1;
    j3 = mod(j2,3) + 1;

    fBatExp = @(i, j, c) (M(sub2ind(size(M), base+i, j)) + c*M(sub2ind(size(M), base+j, i))).*s;
    q( sub2ind([n 4], repmat(find(i), 1, 4), 1+[j1*0 j1 j2 j3]) ) = [fBatExp(j3,j2,-1) r/2 fBatExp(j1, j2, 1) fBatExp(j1, j3, 1)];
end
