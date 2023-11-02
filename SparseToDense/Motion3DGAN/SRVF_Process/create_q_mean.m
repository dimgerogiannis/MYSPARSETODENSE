
function [ q_mean ] = create_q_mean( label )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

addpath ../SRVF_Processing/SRVFcodes/curveframework/
n_samples=0;
L=68;
len=30;  %% target video lenght
LABELS={'Angry_q_mean', 'Contempt_q_mean', 'Disgust_q_mean', 'Fear_q_mean', 'Happy_q_mean', 'Sad_q_mean', 'Surprise_q_mean'};
q_samples={};
x_samples={};

pts_path='C:\Users\daoudi\Desktop\NaimaWorkspace2\Motion3DGAN\Data\COMA_landmarks_neutral2Exp';
%data_path='C:\Users\admin\Desktop\Naima_Workspace\CASIA\CASIA_Aligned_Normalized';
subj_folders=dir(data_path);   %%data_folder=dir(data_path);
subj_folders=subj_folders(3:end);
for j=1:length(subj_folders)
        %current_label= readLabel_CASIA([subj_folders(j).name]);
        %%% avoid contempt expression and DS_Store files
        if subj_folders(j).isdir %&& current_label==label  
           expr_files=dir([data_path  '\' subj_folders(j).name]);          
           expr_files=expr_files(3:end);
           expr_files=natsort({expr_files.name});
           N_frames=length(expr_files);
           % iterate over frames
           Landmarks=zeros(N_frames, 2*L);
           for k=1:N_frames
                name=char(expr_files(k));                
               % pts=dlmread([pts_path '\' data_folder(i).name '\' subj_folders(j).name '\' expr_files(k).name(1:end-4) '_landmarks.txt']); 
               % pts=dlmread([pts_path '\' data_folder(i).name '\' subj_folders(j).name '\' name(1:end-4) '_landmarks.txt']);  
               % Landmarks(k,:)= reshape(pts', 1, 2*L);
               %[pts_path '\' subj_folders(j).name '\' name(1:end-4) '.csv']
                pts=csvread([pts_path '\' subj_folders(j).name '\' name(1:end-4) '.csv'],1,2); 
                pts_x=pts(1:68);
                pts_y=pts(69:end);
                pts=[pts_x;pts_y];
                Landmarks(k,:)= reshape(pts, 1, 2*L);
            end
           Landmarks=Landmarks'; %% size(2*L, F)
           
           if (j==4*12)
              continue,
           end
           X2=ReSampleCurve(Landmarks,len);
           x_samples{end+1}=X2;
           q_samples{end+1}=curve_to_q(X2);
% %            %%%%%%%%%%%%%%%%
%            q=q_samples{end};
%            sqrt(InnerProd_Q(q,q))
% %            save('tstQ.mat', 'q');
% %            return
           n_samples=n_samples+1;
           clear Landmarks;  clear pts;  clear X2; 
        end  
    end 

n_samples

%%%%%% SRVF %%%%%%%
pts_path='C:\Users\admin\Desktop\Naima_Workspace\CK+\CK_landmarks_Normalized';
data_path='C:\Users\admin\Desktop\Naima_Workspace\CK+\CK+_image_aligned_Normalized';
emotion_path='C:\Users\admin\Desktop\Naima_Workspace\CK+\Emotion';
data_folder=dir(data_path);
data_folder=data_folder(3:end);
length(data_folder)
for i=1:length(data_folder)
    %fprintf('CK %d \n',i)
    if i==51
        continue
    end
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    for j=1:length(subj_folders)
        %current_label= readLabel([emotion_path '/' data_folder(i).name '/' subj_folders(j).name(1:3)]);
        %%% avoid contempt expression and DS_Store files
        if subj_folders(j).isdir %&& current_label==label    %&& not(current_label==2)
           expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
           expr_files=expr_files(3:end);
           expr_files=natsort({expr_files.name});
           N_frames=length(expr_files);
           % iterate over frames
           Landmarks=zeros(N_frames, 2*L);
           for k=1:N_frames
               name=char(expr_files(k));
               %pts=dlmread([pts_path '\' data_folder(i).name '\' subj_folders(j).name '\' expr_files(k).name(1:end-4) '_landmarks.txt']); 
               %%[pts_path '\' data_folder(i).name '\' subj_folders(j).name '\' name(1:end-4) '.csv']
               pts=csvread([pts_path '\' data_folder(i).name '\' subj_folders(j).name '\' name(1:end-4) '.csv'],1,2);
               pts_x=pts(1:68);
                pts_y=pts(69:end);
                pts=[pts_x;pts_y];
                Landmarks(k,:)= reshape(pts, 1, 2*L);
           end
           Landmarks=Landmarks'; %% size(2*L, F)
           X2=ReSampleCurve(Landmarks,len);
           x_samples{end+1}=X2;
           q_samples{end+1}=curve_to_q(X2);
           n_samples=n_samples+1;
           clear Landmarks;  clear pts;  clear X2;
        end  
    end 
end
q_mean=Karcher_Mean( q_samples, x_samples, 100,0.9);
fprintf('Karcher_Mean computed') 
%save(['Q_means/' LABELS{label} 'CK_CASIA.mat'], 'q_mean');
save(['Q_means/q_mean_CK_CASIA_data.mat'], 'q_mean');
n_samples
end


function label=readLabel_CASIA(name_dir)
  if strcmp(name_dir(end-8), 'A')
     label=1;
  elseif strcmp(name_dir(end-8), 'D')
     label=3;
  elseif strcmp(name_dir(end-8), 'F')
     label=4;
  elseif strcmp(name_dir(end-8), 'H')
     label=5;
  elseif strcmp(name_dir(end-8), '2')
     label=6;
  elseif strcmp(name_dir(end-8), '1')
     label=7;
  end
end

function label=readLabel(emotion_path)
              d=dir(emotion_path);
              if length(d)< 3
                 label=0;
              else
              fileID1=fopen([emotion_path '/' d(3).name]);
              expression=textscan(fileID1, '%f');
              fclose(fileID1);
              label=expression{1};
              end
              
end


