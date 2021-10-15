loadpath = '//ZEUS/ocs/data/OSNAP/Mooring Data/LOCO mooring central Irminger Sea/LOCO 2011-2018/MMPs/matlab/';
mld2_1 = load(strcat(loadpath,'mld2_1')).MLD_2_1;
mld2_2 = load(strcat(loadpath,'mld2_2')).MLD_2_2;
mld2_3 = load(strcat(loadpath,'mld2_3')).MLD_2_3;
mld2_4 = load(strcat(loadpath,'mld2_4')).MLD_2_4;
mld2_5 = load(strcat(loadpath,'mld2_5')).MLD_2_5;
mld2_6 = load(strcat(loadpath,'mld2_6')).MLD_2_6;
mld2_7 = load(strcat(loadpath,'mld2_7')).MLD_2_7;
MLD2_8 = load(strcat(loadpath,'MLD2_8')).MLD_2_8;
MLD2_9 = load(strcat(loadpath,'MLD2_9')).MLD_2_9;
MLD2_10 = load(strcat(loadpath,'MLD2_10')).MLD_2_10;
MLD2_11 = load(strcat(loadpath,'MLD2_11')).MLD2_11;

%%
MLD = [mld2_1, mld2_2, mld2_3, mld2_4, mld2_5, mld2_6, mld2_7, MLD2_8, MLD2_9, MLD2_10, MLD2_11];

save('../Data/LOCO_MLD.mat','MLD');