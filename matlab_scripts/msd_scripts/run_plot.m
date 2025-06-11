f = 10;                % Frequency in Hz
fs = 1e4;             % Sampling frequency in Hz (must be >> f)
T = 3;                 % Duration in seconds
t = 0:1/fs:T;          % Time vector
y = sin(2*pi*f*t);     % Sinusoidal signal

N = length(y);
Y = fft(y);
res = fs/N;
f = 0:res:fs/2;
Y_mag = abs(Y)/N;
plot(t, y)
plot(f, Y_mag)