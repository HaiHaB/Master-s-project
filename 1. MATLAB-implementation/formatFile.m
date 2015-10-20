function formatFile(CHEMID)

% Get ID, Name, Formula from the web for a specified file
url = strcat('https://www.ebi.ac.uk/chembl/compound/inspect/', CHEMID);
[ID1, Name1, Formula1] = ProcessWeb(url);

%Open the file and overwrite it
fileName = strcat(CHEMID, '.mol');
FID = fopen(fileName, 'r');
i = 1;
tline = fgetl(FID);
A{i} = tline;
while  ischar( tline)
    i = i+1;
    tline = fgetl(FID);
    A{i} = tline;
end
fclose (FID);


%Rewrite file name
FID = fopen(fileName, 'w');
%fprintf (FID, '%-15s %-20s %-20s\n', ID1, Formula1, Name1);
fprintf (FID, '%-20s %-20s %-20s\n',  Formula1, ID1,Name1);
for i = 2: numel(A)
    if A{i+1} == -1
        fprintf(FID, '%s', A{i});
        break;
    else
        fprintf(FID, '%s\n', A{i});
    end
end

fclose (FID);


% Add the ID, Name and Formular to the list of all Compounds
FID1 = fopen('All Compound.txt', 'a');
fprintf (FID1, '%-20s %-20s %-20s\n',  Formula1, ID1,Name1);
fclose (FID1);

end