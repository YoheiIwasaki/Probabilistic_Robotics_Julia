include("../scripts/ideal_robot.jl")

using Distributions
using Random

mutable struct Camera <: AbstractCamera
    map
    lastdata
    distance_range
    direction_range
    distance_noise_rate
    direction_noise
    distance_bias_rate_std
    direction_bias
    phantom_dist
    phantom_prob
    oversight_prob
    occlusion_prob
end

function Camera_(env_map; distance_range=(0.5,6.0), direction_range=(-pi/3, pi/3),
                distance_noise_rate=0.1, direction_noise=pi/90,
                distance_bias_rate_stddev=0.1, direction_bias_stddev=pi/90,
                phantom_prob=0.0, phantom_range_x = (-5.0, 5.0), phantom_range_y = (-5.0, 5.0),
                oversight_prob=0.1, occlusion_prob=0.0)
    self = Camera(ntuple(x->nothing, fieldcount(Camera))...)
    self.map = env_map
    self.lastdata = []
    self.distance_range = distance_range
    self.direction_range  = direction_range
    self.distance_noise_rate = distance_noise_rate
    self.direction_noise = direction_noise
    self.distance_bias_rate_std = rand(Normal(0, distance_bias_rate_stddev))
    self.direction_bias = rand(Normal(0, direction_bias_stddev))
    rx = phantom_range_x
    ry = phantom_range_y
    self.phantom_dist = (Uniform(rx[1], rx[2]), Uniform(ry[1], ry[2]))
    self.phantom_prob = phantom_prob
    self.oversight_prob = oversight_prob
    self.occlusion_prob = occlusion_prob
    return self
end

function noise(self::Camera, relpos)
    ell = rand(Normal(relpos[1], abs(relpos[1])*self.distance_noise_rate))
    phi = rand(Normal(relpos[2],  self.direction_noise))
    return [ell, phi]
end

function bias(self::Camera, relpos)
    return relpos + [relpos[1]*self.distance_bias_rate_std, self.direction_bias]
end

function phantom(self::Camera, cam_pose, relpos)
    if rand(Uniform()) < self.phantom_prob
        pos = [rand(self.phantom_dist[1]), rand(self.phantom_dist[2])]
        return relative_polar_pos(self, cam_pose, pos)
    else
        return relpos
    end
end

function oversight(self::Camera, relpos)
    if rand(Uniform()) < self.oversight_prob
        return nothing
    else
        return relpos
    end
end

function occlusion(self::Camera, relpos)
    if rand(Uniform()) < self.occlusion_prob
        ell = relpos[1] + rand(Uniform())*(self.distance_range[2] - relpos[1])
        phi = relpos[2]
        return [ell, phi]
    else
        return relpos
    end
end

function data(self::Camera, cam_pose)
    observed = []
    for lm in self.map.landmarks
        z = relative_polar_pos(self, cam_pose, lm.pos)
        z = phantom(self, cam_pose, z)
        z = occlusion(self, z)
        z = oversight(self, z)
        if visible(self, z)
            z = bias(self, z)
            z = noise(self, z)
            push!(observed, (z, lm.id))
        end
    end
    self.lastdata = observed
    return observed
end

mutable struct Robot <: AbstractRobot
    pose
    r
    color
    agent
    poses
    sensor
    noise_pdf
    distance_until_noise
    theta_noise
    bias_rate_nu
    bias_rate_omega
    stuck_pdf
    escape_pdf
    time_until_stuck
    time_until_escape
    is_stuck
    kidnap_pdf
    time_until_kidnap
    kidnap_dist
end

function Robot_(pose; agent=nothing, sensor=nothing, color="black", noise_per_meter = 5, noise_std=pi/60,
    bias_rate_stds=(0.1,0.1), expected_stuck_time=1e100, expected_escape_time = 1e-100,
    expected_kidnap_time=1e100, kidnap_range_x=(-5.0, 5.0), kidnap_range_y=(-5.0, 5.0))
    rbt = Robot(ntuple(x->nothing, fieldcount(Robot))...)
    rbt.pose = pose
    rbt.r = 0.2
    rbt.agent = agent
    rbt.sensor = sensor
    rbt.poses = [pose]
    rbt.color = color
    rbt.noise_pdf = Exponential(1.0/(1e-100 + noise_per_meter))
    rbt.distance_until_noise = rand(rbt.noise_pdf)
    rbt.theta_noise = Normal(0, noise_std)
    rbt.bias_rate_nu = rand(Normal(1, bias_rate_stds[1]))
    rbt.bias_rate_omega = rand(Normal(1, bias_rate_stds[2]))
    rbt.stuck_pdf = Exponential(expected_stuck_time)
    rbt.escape_pdf = Exponential(expected_escape_time)
    rbt.time_until_stuck = rand(rbt.stuck_pdf)
    rbt.time_until_escape = rand(rbt.escape_pdf)
    rbt.is_stuck = false
    rbt.kidnap_pdf = Exponential(expected_kidnap_time)
    rbt.time_until_kidnap = rand(rbt.kidnap_pdf)
    rx = kidnap_range_x
    ry = kidnap_range_y
    rbt.kidnap_dist = (Uniform(rx[1], rx[2]), Uniform(ry[1], ry[2]), Uniform(0, 2*pi))
    return rbt
end

function noise(self::Robot, pose, nu, omega, time_interval)
    self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval
    if self.distance_until_noise <= 0.0
        self.distance_until_noise += rand(self.noise_pdf)
        pose[3] += rand(self.theta_noise)
    end
    return pose
end

function bias(self::Robot, nu, omega)
    return nu*self.bias_rate_nu, omega*self.bias_rate_omega
end

function stuck(self::Robot, nu, omega, time_interval)
    if self.is_stuck
        self.time_until_escape -= time_interval
        if self.time_until_escape <= 0.0
            self.time_until_escape += rand(self.escape_pdf)
            self.is_stuck = false
        end
    else
        self.time_until_stuck -= time_interval
        if self.time_until_stuck <= 0.0
            self.time_until_stuck += rand(self.stuck_pdf)
            self.is_stuck = true
        end
    end
    return nu*(!self.is_stuck), omega*(!self.is_stuck)
end

function kidnap(self::Robot, pose, time_interval)
    self.time_until_kidnap -= time_interval
    if self.time_until_kidnap <= 0.0
        self.time_until_kidnap += rand(self.kidnap_pdf)
        return [rand(self.kidnap_dist[1]), rand(self.kidnap_dist[2]), rand(self.kidnap_dist[3])]
    else
        return pose
    end
end

function one_step(self::Robot, time_interval)
    if isnothing(self.agent)
        return
    end
    obs = nothing
    if !isnothing(self.sensor)
        obs = data(self.sensor, self.pose)
    end
    nu, omega = decision(self.agent, obs)
    nu, omega = bias(self, nu, omega)
    nu, omega = stuck(self, nu, omega, time_interval)
    self.pose = state_transition(self, nu, omega, time_interval, self.pose)
    self.pose = noise(self, self.pose, nu, omega, time_interval)
    self.pose = kidnap(self, self.pose, time_interval)
end


if abspath(PROGRAM_FILE) == @__FILE__
    world = World(30, 1.0)
    m = Map()
    append_landmark(m, Landmark([-4.0,2.0]))
    append_landmark(m, Landmark([2.0,-3.0]))
    append_landmark(m, Landmark([3.0,3.0]))
    append(world, m)

    circling = Agent(0.2, 10.0/180*pi)
    r = Robot([0.0, 0.0, 0.0], circling, Camera_(m, occlusion_prob=0.1), "black")
    append(world, r)
    draw(world)

    showanim("test.mp4")
end


