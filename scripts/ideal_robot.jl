using PyPlot
using PyCall
using Printf
@pyimport matplotlib.animation as anim
@pyimport matplotlib.patches as patches

using Base64
function showanim(filename)
    base64_video = base64encode(open(filename))
    display("text/html", """<video controls src="data:video/x-m4v;base64,$base64_video">""")
end

mutable struct World
    objects
    debug
    time_span
    time_interval
    ani
end

World() = World([], false, 10, 1, nothing)
World(debug) = World([], debug, 10, 1, nothing)
World(time_span, time_interval) = World([], false, time_span, time_interval, nothing)

function append(self::World, obj)
    push!(self.objects, obj)
end

function init_draw(self::World, ax)
    ax.set_aspect("equal")
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
end

function draw(self::World)
    fig, ax = subplots()
    init_draw(self, ax)    
    elems = []

    if self.debug
        for i = 1:1000
            one_step(self, i, elems, ax)
        end
    else
        self.ani = anim.FuncAnimation(fig, one_step, fargs=(self, elems, ax), 
            frames = Int(self.time_span/self.time_interval)+1, interval = Int(self.time_interval*1000))
        self.ani[:save]("test.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    end 
    
end

function one_step(i,self::World, elems, ax)
    plt.cla()
    init_draw(self, ax)
    while length(elems) > 0
        pop!(elems)
    end
    vcat(elems, ax.text(-4.4, 4.5, string("t = ", @sprintf("%.2f[s]", self.time_interval * i) ), fontsize=10))
    for obj in self.objects
        draw(obj, ax, elems)
        if applicable(one_step, obj, 1.0)
            one_step(obj, 1.0)
        end
    end
end

mutable struct Agent
    nu
    omega
end
function decision(self::Agent, observation=nothing)
    return self.nu, self.omega
end

abstract type AbstractRobot end

mutable struct IdealRobot <: AbstractRobot
    pose
    r
    color
    agent
    poses
    sensor
end

IdealRobot(pose) = IdealRobot(pose, 0.2, "black", nothing, [pose], nothing)
IdealRobot(pose, color) = IdealRobot(pose, 0.2, color, nothing, [], nothing)
IdealRobot(pose, agent) = IdealRobot(pose, 0.2, "black", agent, [pose], nothing)
IdealRobot(pose, agent, color) = IdealRobot(pose, 0.2, color, agent, [pose], nothing)
IdealRobot(pose, agent, sensor, color) = IdealRobot(pose, 0.2, color, agent, [pose], sensor)

function draw(self::AbstractRobot, ax, elems)
    x,y,theta = self.pose
    xn = x + self.r * cos(theta)
    yn = y + self.r * sin(theta)
    elems = vcat(elems, ax.plot([x, xn], [y,yn], color = self.color))
    c = patches.Circle(xy=(x,y), radius = self.r, fill = false, color = self.color)
    elems = vcat(elems, [ax.add_patch(c)])
    
    push!(self.poses, self.pose)
    elems = vcat(elems, ax.plot([e[1] for e in self.poses], [e[2] for e in self.poses], linewidth=0.5, color="black"))
    
    if !isnothing(self.sensor) && length(self.poses) > 1
        draw(self.sensor, ax, elems, self.poses[length(self.poses)-1])
    end
    
    if !isnothing(self.agent) && applicable(draw, self.agent, ax, elems)
        draw(self.agent, ax, elems)
    end
end

function state_transition(self::AbstractRobot, nu, omega, time)
    t0 = self.pose[3]
    if abs(omega) < 1e-10
        return self.pose + [nu*cos(t0), nu*sin(t0), omega].*time
    else
        return self.pose + [nu/omega*(sin(t0+omega*time)-sin(t0)),
                                  nu/omega*(-cos(t0+omega*time)+cos(t0)),
                                 omega*time]
    end
end

function one_step(self::AbstractRobot, time_interval)
    if isnothing(self.agent)
        return
    end
    obs = nothing
    if !isnothing(self.sensor)
        obs = data(self.sensor, self.pose)
    end
    nu, omega = decision(self.agent, obs)
    self.pose = state_transition(self, nu, omega, time_interval)
end

mutable struct Landmark
    pos
    id
end

Landmark(pos) = Landmark(pos, nothing)

function draw(self::Landmark, ax, elems)
    c = ax.scatter(self.pos[1], self.pos[2], s=100, marker="*", label="landmarks", color="orange")
    elems = vcat(elems, c)
    elems = vcat(elems, ax.text( self.pos[1], self.pos[2] , string("id:", string(self.id)), fontsize=10))
end

mutable struct Map
    landmarks
end

Map() = Map([])

function append_landmark(self::Map, landmark)
    landmark.id = length(self.landmarks)
    push!(self.landmarks, landmark)
end

function draw(self::Map, ax, elems)
    for lm in self.landmarks
        draw(lm, ax, elems)
    end
end

abstract type AbstractCamera end

mutable struct IdealCamera <: AbstractCamera
    map::Map
    lastdata
    distance_range
    direction_range
end

IdealCamera(map) = IdealCamera(map, [], (0.5, 6.0), (-pi/3, pi/3))

function visible(self::AbstractCamera, polarpos)
    if isnothing(polarpos)
        return false
    end
    return self.distance_range[1] <= polarpos[1] <= self.distance_range[2] &&
        self.direction_range[1] <= polarpos[2] <= self.direction_range[2]
        
end

function data(self::AbstractCamera, cam_pose)
    observed = []
    for lm in self.map.landmarks
        p = relative_polar_pos(self, cam_pose, lm.pos)
        if visible(self, p)
            push!(observed, (p, lm.id))
        end
    end
    self.lastdata = observed
    return observed
end

function relative_polar_pos(self::AbstractCamera, cam_pose, obj_pos)
    diff = obj_pos - cam_pose[1:2]
    phi = atan(diff[2], diff[1]) - cam_pose[3]
    while phi > pi
        phi -= 2*pi
    end
    
    while phi < -pi
        phi += 2*pi
    end
    return [hypot(diff[1], diff[2]), phi]
end

function draw(self::AbstractCamera, ax, elems, cam_pose)
    for lm in self.lastdata
        x,y,theta = cam_pose
        distance, direction = lm[1][1], lm[1][2]
        lx = x + distance*cos(direction + theta)
        ly = y + distance*sin(direction + theta)
        elems = vcat(elems, ax.plot([x, lx], [y,ly], color="pink"))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    world = World(10, 1.0)
    m = Map()
    append_landmark(m, Landmark([2,-2]))
    append_landmark(m, Landmark([-1,-3]))
    append_landmark(m, Landmark([3,3]))
    append(world, m)

    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0/180*pi)
    robot1 = IdealRobot([2, 3, pi/6], straight, IdealCamera(m),"black")
    robot2 = IdealRobot([-2, -1, pi/5*6], circling, IdealCamera(m),"red")
    append(world, robot1)
    append(world, robot2)
    draw(world)
    showanim("test.mp4")
end


