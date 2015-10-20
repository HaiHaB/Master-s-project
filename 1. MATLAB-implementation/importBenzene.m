%fid = fopen('CHEMBL277500.mol');
fid = fopen('71-43-2-2d.mol');
nameAndID = fgetl(fid)
somerandomnumbers = fgetl(fid)
copyrightNotice = fgetl(fid)
nAtomsAndBonds = fgetl(fid)

[token, remain] = strtok(nAtomsAndBonds);
numberOfAtoms = str2double(token)
[token, remain] = strtok(remain);
numberOfBonds = str2double(token)

coords = zeros(numberOfAtoms,3);

for i = 1:numberOfAtoms
    
line = fgetl(fid)    

    remain = line
    
    for j = 1:3
        
        [token, remain] = strtok(remain);
        coords(i,j) = str2double(token)

    end

end

for i = 1:numberOfBonds
    
line = fgetl(fid)    

    remain = line
    
    for j = 1:3
        
        [token, remain] = strtok(remain);
        bonds(i,j) = str2double(token)

    end

end

% CALCULATE THE DISTANCES

for i = 1:numberOfBonds
    
   bonds(i,3) = sqrt(sum((coords(bonds(i, 1),:) - coords(bonds(i, 2),:)).^2));
    
    
end

lol = spconvert(bonds);

bonds2 = bonds;
bonds2AZ(:,1:2) = fliplr(bonds(:,1:2))

lol2 = spconvert(bonds2);

connectionMatrix = lol + lol2