function points = process_landmarks( landmarks )


%landmarks=dlmread(path_);
if length(size(landmarks))==2
    landmarks=landmarks(a,[3,1,2]);
end
points=zeros(size(landmarks));
%Centering
for k=1:size(landmarks,1)
    mu_x=mean(landmarks(k,:,1));
    mu_y=mean(landmarks(k,:,2));
    mu_z=mean(landmarks(k,:,3));
    mu=[mu_x, mu_y ,mu_z];
%     fprintf('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
%     size(mu)
%     size(squeeze(landmarks(1,1,:)))
    for j=1:size(landmarks,2)
        landmarks_gram(j,:)=squeeze(landmarks(k,j,:))-mu';
    end
    normFro=sqrt(trace(landmarks_gram*landmarks_gram'));
    land=landmarks_gram/normFro;
    points(k,:,:)=land;
end

end

