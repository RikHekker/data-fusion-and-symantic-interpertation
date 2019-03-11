function [ y ] = normrnd(mu, sigma)
%normrnd sample from a normal distribution with mean mu and
%std sigma

if numel(mu) > 1
    disp('mean must be scalar');
    mu=mu(1);
end

if numel(sigma) > 1
    disp('standard deviation must be scalar');
    sigma=sigma(1);
end

y = mu+sigma*randn;

end

