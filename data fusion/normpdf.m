function [ y ] = normpdf(x, mu, sigma)
%normpdf compute pdf at x using a normal distribution with mean mu and
%variance sigma

if numel(mu) > 1
    disp('mean must be scalar');
    mu=mu(1);
end

if numel(sigma) > 1
    disp('standard deviation must be scalar');
    sigma=sigma(1);
end


y(length(x)) = 0;
for i=1:length(x)
    y (i)= 1 / (sigma*sqrt(2*pi)) * exp( -(x(i)-mu)^2 / (2*sigma^2));
end

end

