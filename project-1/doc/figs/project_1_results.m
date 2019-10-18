clear all; close all; clc

smallIter  = [1 10 100 1000];
smallScore = [-3829 -3817.906673 -3797.541264 -3794.855598];
smallTime  = [3.118679 16.340953 104.396175 1002.670413];

mediumIter  = [1 10 100];
mediumScore = [-42262.863089 -42052.940982 -41960.806839];
mediumTime  = [29.534446 249.303061 2516.823386];

figure
hold on;

subplot(211)
yyaxis left
plot(smallIter, smallScore, 'o-', 'linewidth', 1.5)
ax = gca;
ax.XScale = 'log';
ax.YScale = 'lin';
ylabel('Max. Bayesian Score', 'fontweight', 'bold')

yyaxis right
plot(smallIter, smallTime, 'o-', 'linewidth', 1.5)
grid on; grid minor; box on;
ax = gca;
ax.XScale = 'log';
ax.YScale = 'log';
ylabel('Time [s]', 'fontweight', 'bold')

xlabel('Iterations', 'fontweight', 'bold')
legend('`small` Score', '`small` Time', 'location', 'best')

subplot(212)
yyaxis left
plot(mediumIter, mediumScore, 'x-', 'linewidth', 1.5)
ax = gca;
ax.XScale = 'lin';
ax.YScale = 'log';
ylabel('Max. Bayesian Score', 'fontweight', 'bold')

yyaxis right
plot(mediumIter, mediumTime, 'x-', 'linewidth', 1.5)
grid on; grid minor; box on;
ax = gca;
ax.XScale = 'log';
ax.YScale = 'log';
ylabel('Time [s]', 'fontweight', 'bold')

xlabel('Iterations', 'fontweight', 'bold')
xlim([1 1000])
legend('`medium` Score', '`medium` Time', 'location', 'best')