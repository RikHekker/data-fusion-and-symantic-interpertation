% Calculate likelihood measurement z given particle p
function p_likelihood = getLikelihood(particle, landmarks, measurements, sigma)

if size(landmarks,1) ~= length(measurements)
    disp('Number of measurements does not correspond to number of landmarks!');
end

% Initialize probability to be one
p_likelihood = 1;

% TODO: For each measurement (distance to a single landmark)
for i=1:size(measurements)
    % TODO:  get distance from particle to landmark (predicted measurement)
    distance=sqrt((particle(1)-landmarks(i,1))^2 + (particle(2)-landmarks(i,2))^2);
    
    %        measurement and the measurement standard deviation sigma
    prob_landmark= normpdf(distance, measurements(i),sigma);
end
    % TODO: the measurement likelihood p is product of n_landmark probabilities
    p_likelihood=prod(prob_landmark);

end