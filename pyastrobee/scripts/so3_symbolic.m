%% Rigid body dynamics for SO(3)
% 
% Gives symbolic expressions for entries in the state/control matrices 
% corresponding to orientation and angular velocity
%
% Reference: GuSTO 
% https://github.com/StanfordASL/GuSTO.jl/blob/master/src/scripts/symbolic_math.m
% 
% Note: they use modified rodrigues parameters and body-frame angular
% velocity, so this has been updated to use XYZW quaternions and
% world-frame angular velocity
%
% Note: For SE(3), adding the translational component is simple because this can
% be essentially modified as a double integrator. 
% For reference, in GuSTO they constructed the translation part of the A matrix as 
% A[1:6, 1:6] = kron([0, 1; 0, 0], eye(3)) 
% And for the B matrix, 
% B[4:6,1:3] = Matrix(1.0I,3,3)/model.mass -- translation component
% B[11:13,4:6] = model.Jinv   -- SO(3)
% ^ This inverse inertia tensor result for SO(3) matches what is expected
% from this script 


clear all; clc;
syms Ixx Iyy Izz Ixy Ixz Iyz
assume(Ixx, 'real');
assume(Iyy, 'real');
assume(Izz, 'real');
q = sym('q', [4,1], 'real'); % XYZW quaternion
wg = sym('w', [3,1], 'real'); % Global-frame angular velocity
M = sym('M', [3,1], 'real'); % Moment
I = diag([Ixx,Iyy,Izz]); % Inertia tensor (diagonal only)
% We currently don't have info on Astrobee's off-diagonal inertia values,
% so we'll neglect this for now. Uncomment this if the off-digaonal values
% are known (will give more complex dynamics equations)
% I = [Ixx, Ixy, Ixz; Ixy, Iyy, Iyz; Ixz, Iyz, Izz];

qx = q(1);
qy = q(2);
qz = q(3);
qw = q(4);
GT = [qw, qz, -qy; 
      -qz, qw, qx; 
      qy, -qx, qw; 
      -qx, -qy, -qz];
q_dot = (1/2) * GT * wg; 
wg_dot = inv(I)*(M - cross(wg,I*wg));

% A and B matrices for the rotation component of rigid body motion
Aq = [jacobian(q_dot, [q;wg]); ...
    jacobian(wg_dot, [q;wg])]
Bq = [jacobian(q_dot, [M]); ...
    jacobian(wg_dot, [M])]


%% Output:
% 
% (Re-run the script to confirm these values, but here is the output if
% MATLAB is not easily accessible)
% 
% Aq =
%  
% [     0, -w3/2,  w2/2, w1/2,                   q4/2,                  q3/2,                  -q2/2]
% [  w3/2,     0, -w1/2, w2/2,                  -q3/2,                  q4/2,                   q1/2]
% [ -w2/2,  w1/2,     0, w3/2,                   q2/2,                 -q1/2,                   q4/2]
% [ -w1/2, -w2/2, -w3/2,    0,                  -q1/2,                 -q2/2,                  -q3/2]
% [     0,     0,     0,    0,                      0, (Iyy*w3 - Izz*w3)/Ixx,  (Iyy*w2 - Izz*w2)/Ixx]
% [     0,     0,     0,    0, -(Ixx*w3 - Izz*w3)/Iyy,                     0, -(Ixx*w1 - Izz*w1)/Iyy]
% [     0,     0,     0,    0,  (Ixx*w2 - Iyy*w2)/Izz, (Ixx*w1 - Iyy*w1)/Izz,                      0]
%  
%  
% Bq =
%  
% [     0,     0,     0]
% [     0,     0,     0]
% [     0,     0,     0]
% [     0,     0,     0]
% [ 1/Ixx,     0,     0]
% [     0, 1/Iyy,     0]
% [     0,     0, 1/Izz]
