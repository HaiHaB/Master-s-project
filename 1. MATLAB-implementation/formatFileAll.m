% To read all file in the directory
F = dir('*.mol');

%Override all the previous content
FID = fopen('All Compound.txt', 'w');
fprintf (FID, '%-20s %-20s %-20s\n',  'Chemical Formula', 'Compound ID', ' Compound Name');
fprintf (FID, '------------------------------------------------------------------\n');
fclose(FID);

for i = 1:length(F)
    F(i).name(1:end-4)
    formatFile(F(i).name(1:end-4));
end
