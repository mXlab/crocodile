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
var DEFAULT_FPS = 30 // matches Autolume's rendering cadence -- adjustable at runtime via /crocodile/output/fps
var MIN_FPS = 1
var MAX_FPS = 60
var MIN_TAU = 0.02  // seconds -- floor for both transition and noise-walk time constants
var MAX_TAU = 10
var AUTOLUME_HOST = '127.0.0.1'
var AUTOLUME_PORT = 1338
var AUTOLUME_ADDRESS = '/crocodile/latent/final'

var manifestById = {} // id -> plain array of 512 floats

// w_avg (dataset mean, used as the truncation-trick anchor -- see tick()) and
// perDimStd (per-dimension standard deviation, used to scale the noise walk so it
// matches the real emotion vectors' natural per-dimension variation instead of an
// arbitrary unit scale) are both derived from manifest.csv itself, as a practical
// stand-in for StyleGAN2's own w_avg/W-space statistics (not available to this
// module -- it never touches the actual model, only pre-computed W vectors).
var wAvg = zeros()
var perDimStd = zeros()

function computeManifestStats() {
    var ids = Object.keys(manifestById)
    if (ids.length === 0) {
        wAvg = zeros()
        perDimStd = zeros()
        return
    }
    var sum = zeros()
    for (var i = 0; i < ids.length; i++) {
        var w = manifestById[ids[i]]
        for (var d = 0; d < W_DIM; d++) sum[d] += w[d]
    }
    var mean = sum.map(function (s) { return s / ids.length })
    var variance = zeros()
    for (var j = 0; j < ids.length; j++) {
        var wj = manifestById[ids[j]]
        for (var d2 = 0; d2 < W_DIM; d2++) {
            var diff = wj[d2] - mean[d2]
            variance[d2] += diff * diff
        }
    }
    wAvg = mean
    perDimStd = variance.map(function (v) { return Math.sqrt(v / ids.length) })
    var avgStd = perDimStd.reduce(function (a, b) { return a + b }, 0) / W_DIM
    console.log('[crocodile-control-module] computed w_avg and per-dim std from ' + ids.length + ' manifest vectors (mean per-dim std: ' + avgStd.toFixed(4) + ')')
}

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
        var rowOk = true
        for (var d = 0; d < W_DIM; d++) {
            var val = parseFloat(cols[wStart + d])
            if (!isFinite(val)) {
                console.error('[crocodile-control-module] manifest.csv row id ' + id + ': non-finite value at w_' + d + ', skipping row')
                rowOk = false
                break
            }
            w[d] = val
        }
        if (rowOk) {
            manifestById[id] = w
        }
    }
    console.log('[crocodile-control-module] loaded ' + Object.keys(manifestById).length + ' emotion vectors from manifest.csv')
    computeManifestStats()
}

function zeros() {
    var a = new Array(W_DIM)
    for (var i = 0; i < W_DIM; i++) a[i] = 0
    return a
}

// Gaussian sample via Box-Muller -- used for the noise random walk below, where the
// exact-decay OU discretization assumes normally-distributed innovations.
function gaussianRandom() {
    var u = 0, v = 0
    while (u === 0) u = Math.random()
    while (v === 0) v = Math.random()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

var state = {
    w_u: null,              // 512-array from live_pipeline.py, or null until first message
    current_w: zeros(),     // the "actress" vector, advances toward target_w each tick
    target_w: zeros(),
    target_id: null,
    mode: 'manual',         // 'auto' | 'manual'
    fps: DEFAULT_FPS,       // output rate to Autolume -- also the tick rate everything else runs at
    transitionTau: 0.4,     // seconds -- time constant for current_w's approach to target_w
    running: false,
    mix: 0.5,                // 1 = pure actress, 0 = pure incoming user vector
    noise_amount: 0,         // multiple of each dimension's natural std (see perDimStd) -- 1 = one natural std
    noiseTau: 2,            // seconds -- how quickly the noise random walk wanders (higher = smoother/slower)
    noise_state: zeros(),   // Ornstein-Uhlenbeck walk, ~unit stationary variance; scaled by noise_amount*perDimStd below
    truncationPsi: 0.85,    // StyleGAN-style truncation trick: 1 = off, 0 = collapse to w_avg (see tick())
    has_selected_before: false,
}

var tickTimer = null
function scheduleTick() {
    if (tickTimer) clearInterval(tickTimer)
    tickTimer = setInterval(tick, 1000 / state.fps)
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
    // transition_running has bypass:true (see crocodile-control-panel.json), so its own
    // address never auto-fires on a value change -- including this /SET -- which is what
    // stops it from re-triggering its own outgoing send and flipping forever. Real presses
    // still reach the module via its onTouch script, which sends explicitly (send() ignores
    // bypass).
    receive('/SET', 'transition_running', state.running ? 1 : 0, { clientId: clients[0] })
}

function tick() {
    if (state.target_id === null && state.w_u === null) return // fully idle -- send nothing

    var dt = 1 / state.fps

    if (state.running) {
        // Exact discretization of exponential decay toward target_w with time constant
        // transitionTau (in seconds) -- unlike a plain per-tick blend factor, this stays
        // correct regardless of the tick rate (fps), and is unconditionally stable for
        // any dt/tau ratio.
        var transitionAlpha = 1 - Math.exp(-dt / state.transitionTau)
        for (var i = 0; i < W_DIM; i++) {
            state.current_w[i] += (state.target_w[i] - state.current_w[i]) * transitionAlpha
        }
    }

    // Smooth random walk (Ornstein-Uhlenbeck process, exact discretization) instead of
    // raw filtered white noise -- wanders around zero with time constant noiseTau rather
    // than jittering independently every tick, and stays unconditionally stable for any
    // dt/tau ratio. Stationary variance is normalized to ~1 here; the actual amplitude is
    // applied below as noise_amount * perDimStd[k], i.e. in units of that dimension's own
    // natural variation across the real emotion vectors -- a plain unit-scale multiplier
    // was checked against manifest.csv's actual statistics and found to be ~4x too large
    // at noise_amount=1 (its per-dim std of 1 vs. the data's actual per-dim std of ~0.24),
    // which is why full noise was pushing W so far from any real vector's neighborhood.
    var noiseDecay = Math.exp(-dt / state.noiseTau)
    var noiseDiffusion = Math.sqrt(1 - noiseDecay * noiseDecay)
    for (var j = 0; j < W_DIM; j++) {
        state.noise_state[j] = noiseDecay * state.noise_state[j] + noiseDiffusion * gaussianRandom()
    }

    var w_u = state.w_u || state.current_w
    var final_w = new Array(W_DIM)
    for (var k = 0; k < W_DIM; k++) {
        final_w[k] = state.mix * state.current_w[k]
            + (1 - state.mix) * w_u[k]
            + state.noise_state[k] * state.noise_amount * perDimStd[k]
    }

    // StyleGAN-style truncation trick: pull the composited vector partway back toward
    // w_avg (here, the mean of manifest.csv's own vectors, standing in for the real
    // model's w_avg which this module has no access to). Keeps noise excursions and any
    // already-off-distribution regressor output from drifting into the low-density,
    // artifact-prone regions of W-space -- psi=1 disables this, lower pulls in harder.
    if (state.truncationPsi < 1) {
        for (var m = 0; m < W_DIM; m++) {
            final_w[m] = wAvg[m] + state.truncationPsi * (final_w[m] - wAvg[m])
        }
    }

    send.apply(null, [AUTOLUME_HOST, AUTOLUME_PORT, AUTOLUME_ADDRESS].concat(final_w))
}

module.exports = {

    init: function () {
        loadManifest()
        scheduleTick()
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
            if (state.target_id === null) {
                // no emotion picked yet -- track the visitor 1:1 until the operator acts
                state.current_w = state.w_u.slice()
            }
            return // consumed, not forwarded to widgets
        }

        if (address === '/crocodile/grid/select') {
            var id = args[0]
            var w = manifestById[id]
            if (!w) {
                console.error('[crocodile-control-module] /crocodile/grid/select: unknown id ' + id)
                return
            }
            state.target_id = id
            state.target_w = w
            if (!state.has_selected_before) {
                state.current_w = w.slice()
                state.has_selected_before = true
            }
            if (state.mode === 'auto') {
                state.running = true
            }
            pushFeedback()
            return
        }

        if (address === '/crocodile/transition/mode') {
            state.mode = args[0] === 1 ? 'auto' : 'manual'
            return
        }

        if (address === '/crocodile/transition/time') {
            var timeVal = args[0]
            if (isFinite(timeVal)) {
                state.transitionTau = Math.max(MIN_TAU, Math.min(MAX_TAU, timeVal))
            } else {
                console.error('[crocodile-control-module] /crocodile/transition/time: ignoring non-finite value ' + timeVal)
            }
            return
        }

        if (address === '/crocodile/output/fps') {
            var fpsVal = args[0]
            if (isFinite(fpsVal) && fpsVal > 0) {
                state.fps = Math.max(MIN_FPS, Math.min(MAX_FPS, fpsVal))
                scheduleTick()
            } else {
                console.error('[crocodile-control-module] /crocodile/output/fps: ignoring invalid value ' + fpsVal)
            }
            return
        }

        if (address === '/crocodile/noise/walk_time') {
            var walkTimeVal = args[0]
            if (isFinite(walkTimeVal)) {
                state.noiseTau = Math.max(MIN_TAU, Math.min(MAX_TAU, walkTimeVal))
            } else {
                console.error('[crocodile-control-module] /crocodile/noise/walk_time: ignoring non-finite value ' + walkTimeVal)
            }
            return
        }

        if (address === '/crocodile/output/truncation') {
            var psiVal = args[0]
            if (isFinite(psiVal)) {
                state.truncationPsi = Math.max(0, Math.min(1, psiVal))
            } else {
                console.error('[crocodile-control-module] /crocodile/output/truncation: ignoring non-finite value ' + psiVal)
            }
            return
        }

        if (address === '/crocodile/transition/running') {
            state.running = args[0] === 1
            pushFeedback()
            return
        }

        if (address === '/crocodile/mix/amount') {
            var mixVal = args[0]
            if (isFinite(mixVal)) {
                state.mix = Math.max(0, Math.min(1, mixVal))
            } else {
                console.error('[crocodile-control-module] /crocodile/mix/amount: ignoring non-finite value ' + mixVal)
            }
            return
        }

        if (address === '/crocodile/noise/amount') {
            var noiseVal = args[0]
            if (isFinite(noiseVal)) {
                // in units of each dimension's own natural std across manifest.csv (see
                // perDimStd) -- 1 is already a fairly full-strength wander, headroom to 3
                // is there for deliberately extreme effects, tempered by truncationPsi.
                state.noise_amount = Math.max(0, Math.min(3, noiseVal))
            } else {
                console.error('[crocodile-control-module] /crocodile/noise/amount: ignoring non-finite value ' + noiseVal)
            }
            return
        }

        return data // everything else (session control, status, etc.) passes through unchanged
    },

}
