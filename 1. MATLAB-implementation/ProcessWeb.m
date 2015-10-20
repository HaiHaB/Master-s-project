function [ ID, Name, Formula] = ProcessWeb(url)

%url = 'https://www.ebi.ac.uk/chembl/compound/inspect/CHEMBL282177'
html = urlread(url);
% Use regular expressions to remove undesired markup.
txt = regexprep(html,'<script.*?/script>','');
txt = regexprep(txt,'<style.*?/style>','');
txt = regexprep(txt,'<.*?>','');

% Get first 80 tokens of the txt
remain = txt;
C = cell (1,400);
for i = 1 :400
    [C{i}, remain] = strtok(remain);
end


% Search for the position of 'Compound ID', 'Compound Name' and 'Chemical
% Formula
ind=find(ismember(C,'ID'));
%ind2=find(ismember(C,'Name'));
ind2=find(ismember(C,'Synonyms'));
ind3=find(ismember(C,'Formula'));



% Get the token for Compound ID, Compound Name and Formula
ID = C{find(ismember(C,'ID'))+1};

% Empty name is CHEMBL after that
Name = C{ind2 +1};
if (strcmp(Name, 'Max'))
    %ChEMBL'))
    Name = '';
end
Formula = C{ind3 + 1};


end

%Download text from the web
%http://uk.mathworks.com/matlabcentral/answers/973-how-do-i-read-text-from-html-file