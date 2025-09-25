




clear


pwd_0 = ('\\Tupil\tupil data\#POCUS KIDNEY TRIAL DATA\#POCUS DATA - Shared with TMU');
% % % pwd_0 = ('\\Tupil\tupil data\#POCUS KIDNEY TRIAL DATA\#POCUS DATA - Shared with TMU\NEW');
cd(pwd_0)
fd_name_0 = dir('P*');


for ii = 1:numel(fd_name_0)
    fn_0 = strfind(fd_name_0(ii).name,'_');
    fn0 = str2double(fd_name_0(ii).name(2:fn_0-1));
    ID_ind(ii,1) = fn0;
end



pID = (240:274);



load settings.mat
for pp = pID

    disp(['    ' num2str(pp - pID(1) + 1) ' / ' num2str(numel(pID)) ''])

    Ind_ID = find(ID_ind == pp);


    if ~isempty(Ind_ID)

        fn1 = strfind(fd_name_0(Ind_ID).name,'_');
        pwd_1 = strcat(pwd_0,  '\' , fd_name_0(Ind_ID).name);
        cd(pwd_1)
        fd_name_1 = dir('P*');


        for jj = 1:numel(fd_name_1)
            fn2 = strfind(fd_name_1(jj).name,'_');
            pwd_2 = strcat(pwd_1,  '\' , fd_name_1(jj).name);
            cd(pwd_2)






            pwd_3 = strcat(pwd_2,  '\ROIs');
            if exist(pwd_3)
                cd(pwd_3)



                fn1 = dir('*ROI*.mat');
                fn2 = dir('*segment*.mat');


                delete(fn1.name)
                delete(fn2(1).name)
                delete(fn2(2).name)



            end
        end
    end
end











