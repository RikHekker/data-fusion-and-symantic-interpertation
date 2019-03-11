function z = robot_sim_measure(robot_pose_full, landmarks, sigma)

% Zero mean Gaussian measurement noise
mu = 0;

% Get most recent robot pose
robot_pose = robot_pose_full(:,end);

% For all landmarks
for i=1:size(landmarks,1);
    
    % Calculate distance to landmark
    distance = sqrt((robot_pose(1) - landmarks(i,1))^2 + (robot_pose(2) - landmarks(i,2))^2);
    
    % Disturb measurement with measurement noise
    z(i) = distance + normrnd(mu, sigma);
    
end