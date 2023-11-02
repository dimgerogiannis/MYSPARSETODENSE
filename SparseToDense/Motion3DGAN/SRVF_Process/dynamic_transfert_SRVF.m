function  dynamic_transfer_SRVF(SRVF_file, landmarks_file)
%%%% This function applies the SRVF motion to a given neutral face configuration
%%% For example:
% SRVF_file= '../Data/COMA_SRVF_neutral2Exp/FaceTalk_170725_00137_TA_mouth_up_0_SRVF.mat'
% landmarks_file = '../Data/COMA_landmarks_neutral2Exp/FaceTalk_170725_00137_TA_mouth_up_0.mat'

addpath ./SRVF_Processing/SRVFcodes/curveframework/
addpath ./SRVF_Processing
len=30;
L=68;

 
data=load(SRVF_file);
q_sample=data.q2n;

%%% please change this parameter if the visualization in not good
intensity=0.14;

data=load(landmarks_file);
landmarks=data.coma_landmarks; 
%landmarks=process_landmarks(landmarks);
pts_x=squeeze(landmarks(1,:,1));
pts_y=squeeze(landmarks(1,:,2));
pts_z=squeeze(landmarks(1,:,3));
landmarks=[pts_x;pts_y;pts_z];
Land(:,1)= reshape(landmarks, 1, 3*L);


curve=q_to_curve(q_sample)*intensity;
for h=1:size(curve,2)
      curve(:,h)=curve(:,h)+squeeze(Land(:,1));
end
Land_sequence={};
    for t=1:len        
        T= (reshape(curve(:,t),3,L))';
        Land_sequence{end+1}=T;
%         plot3(T(:,1),T(:,2),T(:,3),'r*');
        plot(T(:,1),T(:,2),'r*');
        T=[];
        pause(0.1)
        
    end
