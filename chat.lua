local model = dofile("model.lua")

-- settings
local MAX_WORDS = 30        -- cap on response length (the model has no end-of-sequence token)
local TEMPERATURE = 0.0     -- 0 = always pick the most likely word; try 0.5-1.0 for variety
local STOP_TOKENS = { ["\n"] = true, ["you:"] = true, ["bot:"] = true }

-- string to table
local function s_to_table(s)
    local chars = {}
    for c in s:gmatch(".") do
        chars[#chars + 1] = c
    end
    return chars
end

-- find element in table
local function find(t, x)
    for i, v in ipairs(t) do
        if v == x then return i end
    end
    return nil
end

-- match word to index in vocabulary
local word_to_idx = {}
for i, w in ipairs(model.vocabulary) do
    word_to_idx[w] = i
end

-- finds the index of the closest word in the vocabulary to the given word
local function closest_word_idx(w)
    if word_to_idx[w] then
        return word_to_idx[w]
    else 
        local best_score = math.huge
        local best_word = ""
        for i, vw in ipairs(model.vocabulary) do
            if vw == "" or vw == "\n" then goto continue end
            local score = 0

            -- add 1 point to score for every character that must be removed
            local st = s_to_table(w)
            local go = s_to_table(vw)
            for k = #st, 1, -1 do
                local c = st[k]
                local idx = find(go, c)
                if not idx then
                    table.remove(st, k)
                    score = score + 1
                else
                    table.remove(go, idx)
                end
            end

            -- add 1 point to score for every character that must be moved
            local target = s_to_table(vw)
            go = st
            for k = 1, #target do
                local c = target[k]
                local idx = find(go, c)
                if idx then
                    if idx ~= 1 then score = score + 1 end
                    table.remove(go, idx)
                end
            end
            
            -- add 1 point to score for every character that must be added
            score = score + math.max(0, #vw - #w)

            -- if this is the best score so far save and move to next word 
            if score < best_score then
                best_score = score
                best_word = vw
            end
            ::continue::
        end
        return word_to_idx[best_word]
    end
end

---------------------------------------------------------------- weights

-- base64 character -> 6-bit value
local b64 = {}
local b64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
for i = 1, #b64_chars do
    b64[b64_chars:byte(i)] = i - 1
end

-- turn {scale, rows, cols, data} (int8 bytes in base64) into a table of rows of floats
local function decode_matrix(m)
    local rows, cols, scale, data = m.rows, m.cols, m.scale, m.data
    local total = rows * cols
    local M = {}
    for r = 1, rows do M[r] = {} end

    local k = 0 -- number of values stored so far
    local function emit(v)
        if k >= total then return end -- ignore base64 padding bytes
        if v > 127 then v = v - 256 end -- unsigned byte -> signed int8
        M[math.floor(k / cols) + 1][k % cols + 1] = v * scale
        k = k + 1
    end

    -- every 4 base64 characters hold 3 bytes
    for i = 1, #data, 4 do
        local a, b, c, d = data:byte(i, i + 3)
        local n = b64[a] * 262144 + b64[b] * 4096 + (b64[c] or 0) * 64 + (b64[d] or 0)
        emit(math.floor(n / 65536))
        emit(math.floor(n / 256) % 256)
        emit(n % 256)
    end
    return M
end

local H = model.hidden_size
local V = #model.vocabulary
local W_ih = decode_matrix(model.weight_ih)     -- (3H, V), rows ordered r, z, n
local W_hh = decode_matrix(model.weight_hh)     -- (3H, H)
local W_out = decode_matrix(model.out_weight)   -- (V, H)
local b_ih = model.bias_ih
local b_hh = model.bias_hh
local b_out = model.out_bias

---------------------------------------------------------------- parsing

-- split on a separator, keeping empty pieces like python's split(" ")
local function split(s, sep)
    local out, start = {}, 1
    while true do
        local i = s:find(sep, start, true)
        if not i then
            out[#out + 1] = s:sub(start)
            break
        end
        out[#out + 1] = s:sub(start, i - 1)
        start = i + #sep
    end
    return out
end

local function parse_line(line)
    line = line:match("^%s*(.-)%s*$"):lower():gsub("[^a-z0-9:\n ]", "")
    return split(line, " ")
end

local function parse_block(block)
    local out = {}
    for _, line in ipairs(split(block, "\n")) do
        local words = parse_line(line)
        if #out > 0 then out[#out + 1] = "\n" end
        for _, w in ipairs(words) do out[#out + 1] = w end
    end
    return out
end

---------------------------------------------------------------- GRU

local function sigmoid(x) return 1 / (1 + math.exp(-x)) end
local function tanh(x) return 2 * sigmoid(2 * x) - 1 end

local function dot(w, h)
    local s = 0
    for i = 1, #h do s = s + w[i] * h[i] end
    return s
end

-- one GRU step for the word with vocabulary index idx
-- the input is one-hot, so W_ih * x is just column idx of W_ih
local function step(idx, h)
    local new_h = {}
    for j = 1, H do
        local jz, jn = H + j, 2 * H + j
        local r = sigmoid(W_ih[j][idx] + b_ih[j] + dot(W_hh[j], h) + b_hh[j])
        local z = sigmoid(W_ih[jz][idx] + b_ih[jz] + dot(W_hh[jz], h) + b_hh[jz])
        local n = tanh(W_ih[jn][idx] + b_ih[jn] + r * (dot(W_hh[jn], h) + b_hh[jn]))
        new_h[j] = (1 - z) * n + z * h[j]
    end
    return new_h
end

-- one score per vocabulary word
local function get_logits(h)
    local out = {}
    for k = 1, V do
        out[k] = dot(W_out[k], h) + b_out[k]
    end
    return out
end

-- choose a word index from the scores
local function pick(logits)
    local mx, best = -math.huge, 1
    for k = 1, V do
        if logits[k] > mx then mx, best = logits[k], k end
    end
    if TEMPERATURE <= 0 then return best end

    local weights, total = {}, 0
    for k = 1, V do
        weights[k] = math.exp((logits[k] - mx) / TEMPERATURE)
        total = total + weights[k]
    end
    local r = math.random() * total
    for k = 1, V do
        r = r - weights[k]
        if r <= 0 then return k end
    end
    return V
end

-- generate a response for the conversation so far
local function reply(history)
    -- build the prompt the same way as in training: every line followed by a newline
    local lines = {}
    for _, l in ipairs(history) do
        lines[#lines + 1] = l .. "\n"
    end
    local prompt = table.concat(lines)

    -- read the prompt to build up the hidden state
    local h = {}
    for j = 1, H do h[j] = 0 end
    for _, w in ipairs(parse_block(prompt)) do
        if not (w == "" and not word_to_idx[""]) then
            h = step(closest_word_idx(w), h)
        end
    end

    -- generate words until a stop token or the length cap
    local logits = get_logits(h)
    local response = {}
    for _ = 1, MAX_WORDS do
        local idx = pick(logits)
        local word = model.vocabulary[idx]
        if STOP_TOKENS[word] then break end
        if word ~= "" then response[#response + 1] = word end
        h = step(idx, h)
        logits = get_logits(h)
    end
    return table.concat(response, " ")
end

---------------------------------------------------------------- chat loop
math.randomseed(os.time())
print("Type 'reset' to clear the conversation, 'quit' to exit.")
local history = {}
while true do
    io.write("> ")
    local input = io.read()
    if not input then break end
    input = input:match("^%s*(.-)%s*$")

    if input:lower() == "quit" then
        break
    elseif input:lower() == "reset" then
        history = {}
        print("(conversation cleared)")
    elseif input ~= "" then
        history[#history + 1] = "you: " .. input
        local response = reply(history)
        print("computer: " .. response)
        history[#history + 1] = "bot: " .. response
    end
end