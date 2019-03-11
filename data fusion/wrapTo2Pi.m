function [ y ] = wrapTo2Pi(x)

y=x;

while y>2*pi
    y = y-2*pi;
end

while y<2*pi
    y = y+2*pi;
end

end

