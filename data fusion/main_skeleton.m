clear
close all
clc

%% SETTINGS

% Number of simulated time steps
n_steps = 20;

% Landmarks measured by the robot, you may try various configurations
%landmarks  = [20 20; 80 80; 20 80; 80 20];
%landmarks  = [20 20; 80 80];
landmarks = [50 50];

% Size of the world
world_width = 100;
world_height = 100;
world = [world_width; world_height];

% Number of particles
N = 1000;

% Plotting options
MSlm = 5;  % Size marker landmarks
MSpar = 5; % Size marker particle

%% Models

% Process model: process noise in each direction
sigma_forward = 0.25;
sigma_turn = 0.1;

% Input
u_forward = 5;
u_turn = 0.05;

% Measurment model: Gaussian measurement noise std
sigma_meas = 3;

%% INITIALIZATION

% Initial robot pose
robot_x = rand(1)*world_width;
robot_y = rand(1)*world_height;
robot_th = rand(1)*2*pi;
robot_pose = [robot_x; robot_y; robot_th];

% Initialize N particles and weights
particles = ones(3,N);  % for efficiency purposes
w = ones(1,N)/N;
for i=1:N
    particles(:,i) = [rand(1)*world_width; rand(1)*world_height; rand(1)*2*pi];
end

%% START SIMULATION
for n=1:n_steps
    
    %% Move the simulated robot
    robot_pose = robot_sim_move(robot_pose, world, u_forward, u_turn);
    
    %% Simulate a measurement
    z = robot_sim_measure(robot_pose, landmarks, sigma_meas);
    
    %% Prediction step: propagate particles using motion model
    for i=1:N
        % TODO: implement prediction step in the particle_move function
        particles(:,i) = particle_move(particles(:,i), world, u_forward, u_turn, sigma_forward, sigma_turn);
    end
    
    % Show propagated particles
    f = showWorld(world, landmarks, 1);
    plot(particles(1,:),particles(2,:),'g.', 'Markersize', MSpar);
    
    %% Calculate likelihood: particle weights
    for i=1:N
        % TODO: calculate measurment likelihood in the getLikelihood
        % function
        w(i) = getLikelihood(particles(:,i), landmarks, z, sigma_meas);
    end
    
    %% Update step: use resampling wheel approach to obtain posterior
    
    % Pick a random particle to start with
    index = max(1,round(rand(1)*N));
    
    % TODO: implement resampling using the resampling wheel
    % > https://www.youtube.com/watch?v=wNQVo6uOgYA
    %   (result: set of new samples)
    beta=0;
    w_max=max(w);
    new_particles=zeros(3,N);
    for i=1:N
      beta=beta+rand(1)*2*w_max;
      while beta > w(index)
          beta= beta-w(index);
          index=index+1;
          if index >N
              index = index-N;
          end
      end
      new_particles(:,i)= particles(:,index);
    end
    particles=new_particles;
      

    
    %% Plot robot pose and pause
    plot(robot_pose(1,end), robot_pose(2,end), 'r-*', 'MarkerSize', 8, 'LineWidth',2);
    hold off
    pause(0.25);
    
end
