
lands_paths='C:\Users\Naima\Desktop\NaimaWorkspace\Codes3D\TF_FLAME-master\data\BU_data\lands3D_Anger\';
d=dir(lands_paths);
d=d(3:end);
addpath ./SRVF_Processing/SRVFcodes/curveframework/


L=68;
len=length(dir(lands_paths))-2; %30;
%data=load('Data/FaceTalk_170913_03279_TA_mouth_side_0.mat');
Landmarks=zeros(len, 3*L);
for k=1:size(30)
    k
        data = load([lands_paths d(k).name])
        landmarks=data.lands; %(30!,68,3)        
        %landmarks=process_landmarks(landmarks);
        pts_x=squeeze(landmarks(:,1));
        pts_y=squeeze(landmarks(:,2));
        pts_z=squeeze(landmarks(:,3));
        pts=[pts_x;pts_y;pts_z];
        Landmarks(k,:)= reshape(pts, 1, 3*L);
end
Landmarks=Landmarks'; %% size(2*L, F)
X2=ReSampleCurve(Landmarks,len);
%x_samples{end+1}=X2;
[q_sample, intensity]=curve_to_q(X2);
% %            %%%%%%%%%%%%%%%%
%            q=q_samples{end};
%            sqrt(InnerProd_Q(q,q))
% %            save('tstQ.mat', 'q');
% %            return
save('./Data/q_anger.mat','q_sample','intensity')
%clear Landmarks;  clear pts;  clear X2; 




