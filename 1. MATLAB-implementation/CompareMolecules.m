function result = CompareMolecules(connectionMatrix1, connectionMatrix2)

NA = size(connectionMatrix1,1);
NB = size(connectionMatrix2,1);

%connectionMatrix1S = sort(connectionMatrix1,2,'descend');
%connectionMatrix2S = sort(connectionMatrix2,2,'descend');
full(connectionMatrix1);
full(connectionMatrix2);

S = zeros(NA, NB);
C = zeros(NA, NB);


for a = 1:NA
   for b = 1:NB
       
         % array of boolean with NB false value
         arrayTemp = false(NB);
        for t = 1:NA
      
            for k = 1:NB
                  if( (arrayTemp(k) == false) && abs(connectionMatrix1(a,t) - connectionMatrix2(b,k))<=0.5)
                    C(a,b)= C(a,b) +1;
                    arrayTemp(k)= true;
                    break;
                  end  
            end
        end
    
    end
    
end
C;
S = C ./ (NA+NB-C);
S

S2 = S;
total = 0;



for i = 1:NA
    
  
    total =  max(max(S2)) + total;
    
    [aMax,bMax] = find(S2 == max(max(S2)),1);
    
    %[aMax,bMax] = ind2sub(2, find(S2 == max(max(S2)), 1))
    
    
    
    S2(aMax,:) = 0;
    S2(:,bMax) =0;
    
    
    
end

result = total / NA;
