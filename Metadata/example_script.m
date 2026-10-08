clear
path = '.\derivatives\sub-01\sub-01_acq-MW2_eeg.mat'; % change path

%% Eye blink data plot
load(path)

event = struct2table(data.EB.event);
task = data.EB.raw;
srate = data.EB.srate;

starttask = event.latency(2); % start task point
fintask = event.latency(23); % end task point

figure
plot(task,'k'); hold on;

for i=1:24
    if(event.edftype(i)==3)
        xl(i) = xline((event.latency(i)), 'b'); 
    else 
        xl(i) = 0;
    end

end

xlim([starttask fintask])
xticks([starttask starttask+srate*10 starttask+srate*20 starttask+srate*30 starttask+srate*40 starttask+srate*50 starttask+srate*60 fintask]);
xticklabels({'0', '10', '20', '30', '40', '50', '60', ''});

ylim([-1000 1000])
yticks([-1000 -500 0 500 1000])
set(gca, 'FontSize', 15)

%% Power spectrum (Eye open & Eye closed)
load(path)

event = struct2table(data.EB.event);
srate = data.EB.srate;

starttask = event.latency(2); % start task point
fintask = event.latency(23); % end task point

MVO_prerest = data.MVO.raw(1:starttask-1);
MVC_prerest = data.MVC.raw(1:starttask-1);
srate = data.MVC.srate;

[EO_p, EO_f] = pspectrum(MVO_prerest,srate);
[EC_p, EC_f] = pspectrum(MVC_prerest,srate);


figure
plot(EO_f, EO_p,'LineWidth',2,'Color','b');
hold on
plot(EC_f, EC_p,'LineWidth',2,'Color','r');
xlim([4 30]);
ylim([0 200]);

set(gca, 'FontSize', 15)

xlabel('Frequency (Hz)','FontSize',15);
ylabel('Log Power (μV^2/Hz)','FontSize',13);

legend(["Eye open", "Eye closed"])


%% Power spectrum (before movement & after movement)
load(path)

event = struct2table(data.EB.event);
srate = data.EB.srate;

starttask = event.latency(2); % start task point
fintask = event.latency(23); % end task point

MVC_prerest = data.MVC.raw(1:starttask-1);
MVC_postrest = data.MVC.raw(fintask:end);
srate = data.MVC.srate;

[pre_p, pre_f] = pspectrum(MVC_prerest,srate);
[post_p, post_f] = pspectrum(MVC_postrest,srate);

figure
plot(pre_f, pre_p,'LineWidth',2,'Color','b');
hold on
plot(post_f, post_p,'LineWidth',2,'Color','r');
xlim([4 30]);
ylim([0 200]);

set(gca, 'FontSize', 15)

ylabel('Log Power (μV^2/Hz)','FontSize',13);
xlabel('Frequency (Hz)','FontSize',15);

legend(["Pre-rest", "Post-rest"])
