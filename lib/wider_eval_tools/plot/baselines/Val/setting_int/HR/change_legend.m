name = {'wider_pr_info_HR_easy_val.mat','wider_pr_info_HR_medium_val.mat','wider_pr_info_HR_hard_val.mat'}
new_legend_name = 'HR(ResNet-101)'
for i=1:length(name)
	load(name{i})
	legend_name = new_legend_name
	save(name{i},'legend_name','pr_cruve')
	
end
