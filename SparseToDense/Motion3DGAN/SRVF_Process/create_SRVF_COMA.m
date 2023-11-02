clear all, 
addpath ./SRVF_Processing/SRVFcodes/curveframework/

L=68; %% Landmarks number
len=30;  %% target video lenght
% here set the path to save SRVF
path_SRVF ='C:\Users\daoudi\Desktop\NaimaWorkspace2\Motion3DGAN\Data\COMA_SRVF_neutral2Exp' ;

data_path='C:\Users\daoudi\Desktop\NaimaWorkspace2\Motion3DGAN\Data\COMA_landmarks_neutral2Exp';

data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
        data=load([data_path '/' data_folder(i).name]);
        Landmarks=zeros(size(data.coma_landmarks,1), 3*L);
        for k=1:size(data.coma_landmarks,1)
            landmarks=data.coma_landmarks; %(30!,68,3)
            landmarks=process_landmarks(landmarks);
            pts_x=squeeze(landmarks(k,:,1));
            pts_y=squeeze(landmarks(k,:,2));
            pts_z=squeeze(landmarks(k,:,3));
            pts=[pts_x;pts_y;pts_z];
            Landmarks(k,:)= reshape(pts, 1, 3*L);
        end
        Landmarks=Landmarks'; 
        X2=ReSampleCurve(Landmarks,len);                     
        [q2n, intensity]=curve_to_q(X2);
        intensity
        sqrt(sum(sum(q2n.*q2n))/size(q2n,2))
        
        clear Landmarks; clear q_sample; clear pts; clear q2n; clear X2;
         
           
end
        


           
           
         
           

            