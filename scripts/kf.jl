include("../scripts/mcl.jl")

function sigma_ellipse(p, cov, n)
    eig_vals= eigvals(cov)
    eig_vec = eigvecs(cov)
    ang = atan(eig_vec[1,:][2], eig_vec[1,:][1])/pi * 180
    return patches.Ellipse(p, width=2*n*sqrt(eig_vals[1]), 
        height=2*n*sqrt(eig_vals[2]), angle=ang, fill=false, color="blue", alpha=0.5)
end

function matM(nu, omega, time, stds)
    return Diagonal([stds["nn"]^2*abs(nu)/time + stds["no"]^2*abs(omega)/time,
            stds["on"]^2*abs(nu)/time + stds["oo"]^2*abs(omega)/time])
end

function matA(nu, omega, time, theta)
    st, ct = sin(theta), cos(theta)
    stw, ctw = sin(theta+omega*time), cos(theta+omega*time)
    return [(stw-st)/omega -nu/(omega^2)*(stw-st)+nu/omega*time*ctw;
                 (-ctw+ct)/omega -nu/(omega^2)*(-ctw+ct)+nu/omega*time*stw;
                 0.0 time]
end

function matF(nu, omega, time, theta)
    F = [1.0 0.0 0.0;0.0 1.0 0.0; 0.0 0.0 1.0]
    F[1, 3] = nu / omega * (cos(theta + omega * time) - cos(theta))
    F[2, 3] = nu / omega * (sin(theta + omega * time) - sin(theta))
    return F
end

function matH(pose, landmark_pos)
    mx, my = landmark_pos
    mux, muy, mut = pose
    q = (mux - mx)^2 + (muy- my)^2
    return [(mux - mx)/sqrt(q) (muy-my)/sqrt(q) 0.0; (my-muy)/q (mux-mx)/q -1.0]
end

function matQ(distance_dev, direction_dev)
    return Diagonal([distance_dev^2, direction_dev^2])
end

mutable struct KalmanFilter
    belief
    pose
    motion_noise_stds
    map
    distance_dev_rate
    direction_dev
end

function KalmanFilter_(envmap, init_pose, motion_noise_stds=Dict("nn"=>0.19, "no"=>0.001, "on"=>0.13, "oo"=>0.2),
                distance_dev_rate=0.14, direction_dev=0.05)
    self = KalmanFilter(ntuple(x->nothing, fieldcount(KalmanFilter))...)
    self.belief = MvNormal([0.0,0.0, 0.0], Diagonal([1e-10, 1e-10, 1e-10]))
    self.pose = self.belief.μ
    self.motion_noise_stds = motion_noise_stds
    self.map = envmap
    self.distance_dev_rate = distance_dev_rate
    self.direction_dev = direction_dev
    return self
end

function motion_update(self::KalmanFilter, nu, omega, time)
    if abs(omega) < 1e-5
        omega = 1e-5
    end
    
    M = matM(nu, omega, time, self.motion_noise_stds)
    A = matA(nu, omega, time, self.belief.μ[3])
    F = matF(nu, omega, time, self.belief.μ[3])
    n_cov = F*self.belief.Σ*F' + A*M*A'
    n_mean = state_transition(IdealRobot(), nu, omega, time, self.belief.μ)
    self.belief = MvNormal(n_mean, Symmetric(n_cov))
    self.pose = self.belief.μ
end

function observation_update(self::KalmanFilter, observation)
    for d in observation
        z = d[1]
        obs_id = d[2]+1 #1indexed
        H = matH(self.belief.μ, self.map.landmarks[obs_id].pos)
        estimated_z = relative_polar_pos(IdealCamera(), self.belief.μ, self.map.landmarks[obs_id].pos)
        Q = matQ(estimated_z[1]*self.distance_dev_rate, self.direction_dev)
        K = self.belief.Σ * H' * inv(Q + H * self.belief.Σ * H')
        n_cov = (Matrix{Float64}(I, 3, 3) - K * H ) * self.belief.Σ
        n_mean = self.belief.μ + K * (z - estimated_z)
        self.belief = MvNormal(n_mean, Symmetric(n_cov))
        self.pose = self.belief.μ
    end
end

function draw(self::KalmanFilter, ax, elems)
    e = sigma_ellipse(self.belief.μ[1:2], self.belief.Σ[1:2, 1:2], 3)
    elems = vcat(elems, [ax.add_patch(e)])
    
    x,y,c = self.belief.μ
    sigma3 = sqrt(self.belief.Σ[3,3])*3
    xs = [x + cos(c-sigma3), x, x + cos(c+sigma3)]
    ys = [y + sin(c-sigma3), y, y + sin(c+sigma3)]
    elems = vcat(elems, ax.plot(xs, ys, color="blue", alpha=0.5))
end

# function trial()
#     time_interval = 1.0
#     world = World_(30, time_interval, false)
    
#     m = Map()
#     for ln in [[-4.0, 2.0], [2.0, -3.0], [3.0, 3.0]]
#         append_landmark(m, Landmark(ln))
#     end
#     append(world, m)
    
#     initial_pose = [0.0, 0.0, 0.0]
#     kf = KalmanFilter_(m, initial_pose)
#     circling = EstimationAgent_(time_interval, 0.2, 10.0/180*pi, kf)
#     r = Robot_(initial_pose, sensor=Camera_(m), agent=circling, color="red")
#     append(world, r)
    
#     kf2 = KalmanFilter_(m, initial_pose)
#     linear = EstimationAgent_(time_interval, 0.1, 0.0, kf2)
#     r2 = Robot_(initial_pose, sensor=Camera_(m), agent=linear, color="red")
#     append(world, r2)
    
#     kf3 = KalmanFilter_(m, initial_pose)
#     right = EstimationAgent_(time_interval, 0.1, -3.0/180*pi, kf3)
#     r3 = Robot_(initial_pose, sensor=Camera_(m), agent=right, color="red")
#     append(world, r3)

#     draw(world)
#     showanim("test.mp4")
# end

# trial()


