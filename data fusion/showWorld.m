function f = showWorld(world, landmarks,j)

f = figure(j);
for i=1:size(landmarks,1)
    plot(landmarks(i,1), landmarks(i,2), 'ko', 'MarkerSize', 5, 'Linewidth', 5);
    hold on;
end
xlim([0, world(1)]);
ylim([0, world(2)]);