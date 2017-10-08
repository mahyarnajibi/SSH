function plot_pr(propose,recall,lendge_name,seting_class,setting_name,dateset_class,plot_out_path)
model_num = size(propose,1);
figure1 = figure('PaperSize',[20.98 29.68],'Color',[1 1 1], 'rend','painters','pos',[1 1 800 400]);
axes1 = axes('Parent',figure1,...
    'LineWidth',2,...
    'FontSize',10,...
    'FontName','Times New Roman',...
    'FontWeight','bold');
box(axes1,'on');
hold on;

LineColor = colormap(hsv(model_num));
for i=1:model_num
    plot(propose{i},recall{i},...
        'MarkerEdgeColor',LineColor(i,:),...
        'MarkerFaceColor',LineColor(i,:),...
        'LineWidth',3,...
        'Color',LineColor(i,:))
    grid on;
    hold on;
end
legend1 = legend(lendge_name,'show');
set(legend1,'Location','EastOutside');

xlim([0,1]);
ylim([0,1]);
xlabel('Recall');
ylabel('Precision');
savedir = plot_out_path;
if ~exist(savedir)
    mkdir(savedir);
end
savename = fullfile(savedir,sprintf('wider_pr_curve_%s_%s.pdf',seting_class,setting_name));
saveTightFigure(gcf,savename);
clear gcf;
hold off;




