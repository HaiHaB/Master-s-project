function formatFileAllAnotherDirectory(DIREC)

% To read all file in the directory
directory = strcat(DIREC,'*.mol');
F = dir(directory);

%Override all the previous content
FID = fopen('All Compound.txt', 'w');
fprintf (FID, '%-20s %-20s %-20s\n',  'Chemical Formula', 'Compound ID', ' Compound Name');
fprintf (FID, '------------------------------------------------------------------\n');
fclose(FID);

length(F)
for i = 1:length(F)
    formatFileAnotherDirectory(DIREC,F(i).name(1:end-4));
    pause(0.01);
end
end
