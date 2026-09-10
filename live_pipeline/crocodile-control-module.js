// Custom Open Stage Control module for the live latent controller.
// Loaded via `open-stage-control --custom-module live_pipeline/crocodile-control-module.js`
// (see run_control_panel.sh). Holds all latent-vector state and math -- the
// session JSON's widgets only send lightweight control messages here and
// receive small feedback pushes back; the 512-float vectors themselves never
// round-trip through widget values.
//
// See docs/superpowers/specs/2026-09-10-live-latent-controller-design.md.

var fs = nativeRequire('fs')
var path = nativeRequire('path')

var MANIFEST_PATH = path.join(__dirname, '..', 'emotion_grid', 'data', 'manifest.csv')
var W_DIM = 512
var TICK_MS = 1000 / 30 // 30 Hz, matches Autolume's rendering cadence
var NOISE_SMOOTHING = 0.05 // lower = slower-drifting noise
var AUTOLUME_HOST = '127.0.0.1'
var AUTOLUME_PORT = 1338
var AUTOLUME_ADDRESS = '/crocodile/latent/final'

var manifestById = {} // id -> plain array of 512 floats

function loadManifest() {
    manifestById = {}
    var text
    try {
        text = fs.readFileSync(MANIFEST_PATH, 'utf8')
    } catch (e) {
        console.error('[crocodile-control-module] could not read manifest.csv at ' + MANIFEST_PATH + ': ' + e.message)
        return
    }
    var lines = text.split('\n').filter(function (l) { return l.trim().length > 0 })
    if (lines.length < 2) {
        console.error('[crocodile-control-module] manifest.csv has no data rows: ' + MANIFEST_PATH)
        return
    }
    var header = lines[0].split(',')
    var idCol = header.indexOf('id')
    var wStart = header.indexOf('w_000')
    if (idCol === -1 || wStart === -1) {
        console.error('[crocodile-control-module] manifest.csv missing id/w_000 columns')
        return
    }
    for (var i = 1; i < lines.length; i++) {
        var cols = lines[i].split(',')
        var id = cols[idCol]
        var w = new Array(W_DIM)
        for (var d = 0; d < W_DIM; d++) {
            w[d] = parseFloat(cols[wStart + d])
        }
        manifestById[id] = w
    }
    console.log('[crocodile-control-module] loaded ' + Object.keys(manifestById).length + ' emotion vectors from manifest.csv')
}

function zeros() {
    var a = new Array(W_DIM)
    for (var i = 0; i < W_DIM; i++) a[i] = 0
    return a
}

var state = {
    w_u: null,              // 512-array from live_pipeline.py, or null until first message
    current_w: zeros(),     // the "actress" vector, advances toward target_w each tick
    target_w: zeros(),
    target_id: null,
    mode: 'manual',         // 'auto' | 'manual'
    speed: 0.08,
    running: false,
    mix: 0.5,                // 1 = pure actress, 0 = pure incoming user vector
    noise_amount: 0,
    noise_state: zeros(),
}

var clients = []

app.on('open', function (data, client) {
    if (clients.indexOf(client.id) === -1) clients.push(client.id)
})
app.on('close', function (data, client) {
    var idx = clients.indexOf(client.id)
    if (idx !== -1) clients.splice(idx, 1)
})

function pushFeedback() {
    if (clients.length === 0) return
    receive('/SET', 'selected_id', state.target_id || '', { clientId: clients[0] })
    receive('/SET', 'transition_running', state.running ? 1 : 0, { clientId: clients[0] })
}

function tick() {
    if (state.target_id === null) return // nothing selected yet -- send nothing

    if (state.running) {
        for (var i = 0; i < W_DIM; i++) {
            state.current_w[i] += (state.target_w[i] - state.current_w[i]) * state.speed
        }
    }

    for (var j = 0; j < W_DIM; j++) {
        var white = Math.random() * 2 - 1
        state.noise_state[j] += (white - state.noise_state[j]) * NOISE_SMOOTHING
    }

    var w_u = state.w_u || state.current_w
    var final_w = new Array(W_DIM)
    for (var k = 0; k < W_DIM; k++) {
        final_w[k] = state.mix * state.current_w[k]
            + (1 - state.mix) * w_u[k]
            + state.noise_state[k] * state.noise_amount
    }

    send.apply(null, [AUTOLUME_HOST, AUTOLUME_PORT, AUTOLUME_ADDRESS].concat(final_w))
}

module.exports = {

    init: function () {
        loadManifest()
        setInterval(tick, TICK_MS)
    },

    oscInFilter: function (data) {
        var address = data.address
        // NOTE: verified empirically (Step 2) -- data.args elements are
        // {value, type} objects, not plain numbers/strings. Extract .value
        // before use.
        var args = data.args.map(function (a) { return a.value })

        if (address === '/crocodile/latent/user') {
            if (args.length !== W_DIM) {
                console.error('[crocodile-control-module] /crocodile/latent/user: expected ' + W_DIM + ' floats, got ' + args.length)
                return
            }
            state.w_u = args.slice()
            return // consumed, not forwarded to widgets
        }

        if (address === '/grid/select') {
            var id = args[0]
            var w = manifestById[id]
            if (!w) {
                console.error('[crocodile-control-module] /grid/select: unknown id ' + id)
                return
            }
            state.target_id = id
            state.target_w = w
            if (state.mode === 'auto') {
                state.running = true
            }
            pushFeedback()
            return
        }

        if (address === '/transition/mode') {
            state.mode = args[0] === 1 ? 'auto' : 'manual'
            return
        }

        if (address === '/transition/speed') {
            state.speed = args[0]
            return
        }

        if (address === '/transition/running') {
            state.running = args[0] === 1
            pushFeedback()
            return
        }

        if (address === '/mix/amount') {
            state.mix = args[0]
            return
        }

        if (address === '/noise/amount') {
            state.noise_amount = args[0]
            return
        }

        return data // everything else (session control, status, etc.) passes through unchanged
    },

}
