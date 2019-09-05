function tmax = maxInjectiveStepSizeForMetrics(m, m2, tol)

if nargin<3, tol = 1e-12; end

d = m2 - m;

%% code from AQP implementation: computeInjectiveStepSize.m
a = (d(:,1).*d(:,4) - d(:,2).^2).*d(:,6) - d(:,1).*d(:,5).^2 - d(:,3).^2.*d(:,4) + 2*d(:,2).*d(:,3).*d(:,5);
b = -d(:,3).^2.*m(:,4) + (d(:,4).*m(:,6)+d(:,6).*m(:,4)).*d(:,1) + (d(:,4).*d(:,6) - d(:,5).^2).*m(:,1) - (d(:,2).*m(:,6) + 2 * d(:,6).*m(:,2)).*d(:,2) ...
    + 2*(-d(:,1).*m(:,5) + d(:,2).*m(:,3) + d(:,3).*m(:,2)).*d(:,5) + 2*(d(:,2).*m(:,5) - d(:,4).*m(:,3)).*d(:,3);

c = -d(:,4).*m(:,3).^2 + (d(:,4).*m(:,6) + d(:,6).*m(:,4)).*m(:,1) -(2*d(:,2).*m(:,6)+d(:,6).*m(:,2)).*m(:,2) + (m(:,4).*m(:,6) - m(:,5).^2).*d(:,1) ...
    + 2*(d(:,2).*m(:,3) + d(:,3).*m(:,2) - d(:,5).*m(:,1)).*m(:,5) + 2*(-d(:,3).*m(:,4) + d(:,5).*m(:,2)).*m(:,3);

d = (m(:,1).*m(:,4) - m(:,2).^2).*m(:,6) - m(:,1).*m(:,5).^2 + (2 * m(:,2).*m(:,5) - m(:,3).*m(:,4)).*m(:,3);


%% regular case - cubic equation
delta0 = b.^2-3.*a.*c;
delta1 = 2.*b.^3 - 9.*a.*b.*c + 27.*a.^2.*d;
C=((delta1+sqrt(delta1.^2-4*delta0.^3))/2).^(1/3);
u1 = 1;
u2 = ((-1+sqrt(3)*sqrt(-1))/2);
u3 = ((-1-sqrt(3)*sqrt(-1))/2);
%
ti1 = -(b+u1*C+delta0./(u1*C))./(3*a);
ti2 = -(b+u2*C+delta0./(u2*C))./(3*a);
ti3 = -(b+u3*C+delta0./(u3*C))./(3*a);

%% quadratic equation
ff = (abs(a)<=tol);
ti1(ff) = (-c(ff)+sqrt(c(ff).^2-4*b(ff).*d(ff)))/2./b(ff);
ti2(ff) = (-c(ff)-sqrt(c(ff).^2-4*b(ff).*d(ff)))/2./b(ff);
ti3(ff) = ti2(ff);

%% linear equation
ff = (abs(a)<=tol & abs(b)<=tol);
ti1(ff) = -d(ff)./c(ff);
ti2(ff) = ti1(ff);
ti3(ff) = ti1(ff);

%% deal with pseudo complex roots
ff=(abs(imag(ti1))<=tol);
ti1(ff) = real(ti1(ff));
%
ff=(abs(imag(ti2))<=tol);
ti2(ff) = real(ti2(ff));
%
ff=(abs(imag(ti3))<=tol);
ti3(ff) = real(ti3(ff));

%% get rid of negatives/imaginaries
ti1(ti1<0 | imag(ti1)~=0) = inf;
ti2(ti2<0 | imag(ti2)~=0) = inf;
ti3(ti3<0 | imag(ti3)~=0) = inf;

%% take minimal value
tmax = min( [min(ti1) min(ti2) min(ti3)] );