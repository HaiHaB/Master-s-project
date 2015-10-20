function CompareEachCompound(name)

[MatrixToBeCompared, NameToBeCompared] = parsedFile(name);

% To read every file
F = dir('*.mol');

%Names = cell ( length(F),1);
Fomulars = cell (length(F),1);
Similarity = zeros(length(F),1);

for i = 1:length(F)
    %  fid = fopen(F(ii).name);
    [connectionMatrix2, fomular] = parsedFile(F(i).name);
    % Names{i} = F(i).name;
    Fomulars{i} = fomular
    Similarity(i) = CompareMolecules(MatrixToBeCompared, connectionMatrix2);
    %[result] = CompareMolecules(MatrixToBeCompared, connectionMatrix2);
end



%Print to a file the result
%fileID = fopen('result.txt', 'w');
newName = strcat(name(1:end-4), '.txt');
fileID = fopen(newName, 'w');
fprintf(fileID,'Molecule %s (%s): \n\n', (strtok(NameToBeCompared)),name(1:end-4));

%fprintf(fileID,'%-5s %-20s %-20s %-10s \n','Rank',  'Label', 'Fomulars', 'Similarity score' );
fprintf(fileID,'%-5s %-20s %-20s %-20s %-20s \n','Rank',  'Similarity score', 'Chemical Formula', 'Compound ID', 'Compound Name' );
[Sorted,I] = sortrows(Similarity,-1);

for t = 1: length(I)
    %   fprintf(fileID, '%-5.0f %-20s %-20s %-10.4f \n',t,  Names{I(t)}(1:end-4), Fomulars{I(t)}, Sorted(t));
    fprintf(fileID, '%-5.0f %-20.4f %-60s  \n',t,  Sorted(t), Fomulars{I(t)});
   % Sorted(t)
  %  Fomulars{I(t)}
end

fclose(fileID);

end