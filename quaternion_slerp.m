function q = quaternion_slerp(q1, q2, t, normalized)

% https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp

fNormalize = @(x) x./sqrt(sum(x.^2,2));

if nargin<4 || ~normalized
    q1 = fNormalize(q1);
    q2 = fNormalize(q2);
end

d = sum(q1.*q2, 2);

i = d<0;
d(i) = -d(i);
q2(i, :) = -q2(i, :);

% DOT_THRESHOLD = 0.999;
% i = d>DOT_THRESHOLD;
% q = q1;
% q(i,:)  = fNormalize( q1 + t*(q2-q1) );

d = min(d, 1);

theta0 = acos(d);
theta = theta0*t;

sin_theta = sin(theta);
sin_theta0 = sin(theta0);

s1 = cos(theta) - d .* sin_theta ./ sin_theta0;
s2 = sin_theta ./ sin_theta0;

i = abs(sin_theta0)<1e-6;
s1(i) = 0; s2(i) = 1;

q = s1.*q1 + s2.*q2;