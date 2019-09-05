function d = symmetricDirichlet(x, y, tets)
    J = meshJacobians(x, y, uint64(tets-1));
    fBlockNorm = @(m) sum(reshape(m', 9, [] ).^2)';
    d = fBlockNorm(J) + fBlockNorm(block_inverse(J));
end

function invA = block_inverse(A)

invA = A*0;
inc = (0:size(A,1)/3-1)*3;
invA(1+inc, 1) = A(2+inc,2).*A(3+inc,3) - A(2+inc,3).*A(3+inc,2);
invA(1+inc, 2) = A(1+inc,3).*A(3+inc,2) - A(1+inc,2).*A(3+inc,3);
invA(1+inc, 3) = A(1+inc,2).*A(2+inc,3) - A(1+inc,3).*A(2+inc,2);

invA(2+inc, 1) = A(2+inc,3).*A(3+inc,1) - A(2+inc,1).*A(3+inc,3);
invA(2+inc, 2) = A(1+inc,1).*A(3+inc,3) - A(1+inc,3).*A(3+inc,1);
invA(2+inc, 3) = A(1+inc,3).*A(2+inc,1) - A(1+inc,1).*A(2+inc,3);

invA(3+inc, 1) = A(2+inc,1).*A(3+inc,2) - A(2+inc,2).*A(3+inc,1);
invA(3+inc, 2) = A(1+inc,2).*A(3+inc,1) - A(1+inc,1).*A(3+inc,2);
invA(3+inc, 3) = A(1+inc,1).*A(2+inc,2) - A(1+inc,2).*A(2+inc,1);

detA = invA(1+inc, 1).*A(1+inc, 1) + invA(1+inc, 2).*A(2+inc, 1) + invA(1+inc, 3).*A(3+inc, 1);
invA = invA./reshape( [detA detA detA]', [], 1 );

end