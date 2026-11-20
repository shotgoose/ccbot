local x
local y

local width = 10
local height = 18

local x0 = 0
local y0 = 2
local trees = {}
local dests = {}

for j = 0, height do
    for i = 0, width do
        if (i - x0) % 3 == 0 and (j - y0) % 3 == 0 then
            spots[#spots + 1] = {i, j}
            dests[#dests + 1] = {i, j - 1}
        end
    end
end

local function forward(blocks)
    for i = 0, blocks do
        turtle.forward()
        y = y + 1
    end
end

local function back(blocks)
    for i = 0, blocks do
        turtle.back()
        y = y - 1
    end
end

local function right(blocks)
    turtle.turnRight()
    for i = 0, blocks do
        turtle.forward()
        x = x + 1
    end
    turtle.turnLeft()
end

local function left(blocks)
    turtle.turnLeft()
    for i = 0, blocks do
        turtle.forward()
        x = x - 1
    end
    turtle.turnRight()
end

local function start()
    forward()
end

local function inspect()
    local detect, front = turtle.inspect()
    if not detect then return false end
    local tags = front["tags"]
    if tags["minecraft:saplings"] then
        return "sapling"
    elseif tags["minecraft:logs"] then
        return "log"
    end
    return "unknown"
end

local function main()
    while true do
        local block = inspect()
        if block then
            
        end
    end
end