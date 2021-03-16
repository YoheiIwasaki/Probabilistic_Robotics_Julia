include("../scripts/robot.jl")

using Distributions
using LinearAlgebra
using StatsBase

mutable struct Particle
    pose
    weight
end

function motion_update(self::Particle, nu, omega, time, noise_rate_pdf)
    ns = rand(noise_rate_pdf)
    noised_nu = nu + ns[1]*sqrt(abs(nu)/time) + ns[2]*sqrt(abs(omega)/time)
    noised_omega = omega + ns[3]*sqrt(abs(nu)/time) + ns[4]*sqrt(abs(omega)/time)
    self.pose = state_transition(IdealRobot([]), noised_nu, noised_omega, time, self.pose)
end

function observation_update(self::Particle, observation, envmap, distance_dev_rate, direction_dev)
#     print(observation)
    for d in observation
        obs_pos = d[1]
        obs_id = d[2]+1#1-indexed in julia
        
        pos_on_map = envmap.landmarks[obs_id].pos
        particle_suggest_pos = relative_polar_pos(IdealCamera(), self.pose, pos_on_map)
        
        distance_dev = distance_dev_rate*particle_suggest_pos[1]
        cov = Diagonal([distance_dev^2, direction_dev^2])
        self.weight *= pdf(MvNormal(particle_suggest_pos, cov), obs_pos)
    end
end

mutable struct Mcl
    particles
    map
    distance_dev_rate
    direction_dev
    motion_noise_rate_pdf
    ml
    pose
end

function Mcl_(envmap, init_pose, num, motion_noise_stds=Dict("nn"=>0.19, "no"=>0.001, "on"=>0.13, "oo"=>0.2),
                        distance_dev_rate=0.14, direction_dev=0.05)
    self = Mcl(ntuple(x->nothing, fieldcount(Mcl))...)
    self.particles =[Particle(init_pose, 1.0/num) for i in 1:num-1]
    self.map = envmap
    self.distance_dev_rate = distance_dev_rate
    self.direction_dev = direction_dev
    v = motion_noise_stds
    c = Diagonal([v["nn"]^2, v["no"]^2, v["on"]^2, v["oo"]^2])
    self.motion_noise_rate_pdf = MvNormal(zeros(4), c)
    self.ml = self.particles[1]
    self.pose = self.ml.pose
    return self
end

function set_ml(self::Mcl)
    i = argmax([p.weight for p in self.particles])
    self.ml = self.particles[i]
    self.pose = self.ml.pose
end

function motion_update(self::Mcl, nu, omega, time)
    for p in self.particles
        motion_update(p, nu, omega, time, self.motion_noise_rate_pdf)
    end
end

function observation_update(self::Mcl, observation)
    for p in self.particles
        observation_update(p, observation, self.map, self.distance_dev_rate, self.direction_dev)
    end
    set_ml(self)
    resampling(self)
end

rand(Uniform(0,10))

x = []
append!(x, 1)

function resampling(self::Mcl)
    ws = cumsum([e.weight for e in self.particles])
    if ws[end] < 1e-100
        ws = [e+1e-100 for e in ws]
    end
    step = ws[end]/length(self.particles)
    r = rand(Uniform(0.0, step))
    cur_pos = 1
    ps = []
    while (length(ps) < length(self.particles))
        if r < ws[cur_pos]
            push!(ps, self.particles[cur_pos])
            r += step
        else
            cur_pos += 1
#             if cur_pos > length(self.particles)
#                 cur_pos = 1
#             end
        end
        
    end
    self.particles = [deepcopy(e) for e in ps]
    for p in self.particles
        p.weight = 1.0/length(self.particles)
    end
end

function draw(self::Mcl, ax, elems)
    xs = [p.pose[1] for p in self.particles]
    ys = [p.pose[2] for p in self.particles]
    vxs = [cos(p.pose[3])*p.weight*length(self.particles) for p in self.particles]
    vys = [sin(p.pose[3])*p.weight*length(self.particles) for p in self.particles]
    elems = vcat(elems, ax.quiver(xs, ys, vxs, vys, angles="xy", scale_units="xy", color="blue", alpha=0.5))
end

mutable struct EstimationAgent <: AbstractAgent
    nu
    omega
    estimator
    time_interval
    prev_nu
    prev_omega
end

function  EstimationAgent_(time_interval, nu, omega,estimator)
    self = EstimationAgent(ntuple(x->nothing, fieldcount(EstimationAgent))...)
    self.nu = nu
    self.omega = omega
    self.estimator = estimator
    self.time_interval = time_interval
    self.prev_nu = 0.0
    self.prev_omega = 0.0
    return self
end

function decision(self::EstimationAgent, observation=nothing)
    motion_update(self.estimator, self.prev_nu, self.prev_omega, self.time_interval)
    self.prev_nu, self.prev_omega = self.nu, self.omega
    observation_update(self.estimator,observation)
    return self.nu, self.omega
end

function draw(self::EstimationAgent, ax, elems)
    draw(self.estimator, ax, elems)
    x,y,t = self.estimator.pose
    s = @sprintf("(%.2f, %.2f, %d)", x, y, Int(round(t*180/pi))%360)
    elems = vcat(elems, ax.text(x, y+0.1, s, fontsize=8))
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
#     estimator = Mcl_(m, initial_pose, 100)
#     a = EstimationAgent_(time_interval, 0.2, 10.0/180*pi, estimator)
#     r = Robot_(initial_pose, sensor=Camera_(m), agent=a, color="red")
#     append(world, r)
#     draw(world)
#     showanim("test.mp4")
# end

# if abspath(PROGRAM_FILE) == @__FILE__
#     trial()
# end


