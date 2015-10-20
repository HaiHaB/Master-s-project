function CompareAll()
tic
J = dir('*.mol');
for j = 1: length(J) 
    CompareEachCompound(J(j).name);
end
toc
end

