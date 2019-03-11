function full_robot_pose = robot_sim_move(full_robot_pose, world, forward, turn)

% All noise signals are zero mean Gaussians
mu = 0;
sigma_forward = 0.5;
sigma_turn = 0.1;

% Get most recent robot pose
robot_pose = full_robot_pose(:,end);

% Update the orientation
robot_pose(3) = wrapTo2Pi(robot_pose(3) + turn + normrnd(mu, sigma_turn));

% Distance robot has to move
distance = forward + normrnd(mu, sigma_forward);

% Update (x,y)-position
robot_pose(1) = robot_pose(1) + cos(robot_pose(3)) * distance;
robot_pose(2) = robot_pose(2) + sin(robot_pose(3)) * distance;

% Make sure the robot stays inside the (cyclic) world
for i=1:2
    robot_pose(i) = mod(robot_pose(i), world(i));
end

full_robot_pose = [full_robot_pose robot_pose];