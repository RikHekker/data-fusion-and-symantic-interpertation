function particle = particle_move(particle, world, forward, turn, sigma_forward, sigma_turn);

% Check input
if length(particle) ~= 3
    disp(['Particle p has size ', length(particle), ', but should be 3 (x,y,th)']);
end

% TODO: draw noise sample for the forward motion (zero mean)
noise_forward= normrnd(0,sigma_forward);
% TODO: draw noise sample for the angular motion (zero mean)
noise_turn= normrnd(0,sigma_turn);
% TODO: update particle (particle(1): x, particle(2): y, particle(3): theta)
radtot=particle(3)+wrapTo2Pi(noise_turn+turn);
rad=particle(3)+wrapTo2Pi(turn);
particle=[particle(1)+(noise_forward+forward)*cos(rad); particle(2)+(noise_forward+forward)*sin(rad); radtot];
% Make sure the particle stays inside the (cyclic) world
for i=1:2
    particle(i) = mod(particle(i), world(i));
end
