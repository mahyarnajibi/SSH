function wider_csv(dataset_path,split,output_file)
split_data = load([dataset_path '/wider_face_split/wider_face_' split '.mat']);
fhandle = fopen(output_file,'w');
for i=1:length(split_data.event_list)
    for j=1:length(split_data.file_list{i})
        for k=1:size(split_data.face_bbx_list{i}{j},1)
        fprintf(fhandle,'%s/%s.jpg,%d,%d,%d,%d\n',split_data.event_list{i},...
            split_data.file_list{i}{j},round(split_data.face_bbx_list{i}{j}(k,1)),...
            round(split_data.face_bbx_list{i}{j}(k,2)),round(split_data.face_bbx_list{i}{j}(k,3)),...
            round(split_data.face_bbx_list{i}{j}(k,4)));
        end
    end
end
fclose(fhandle);
end
