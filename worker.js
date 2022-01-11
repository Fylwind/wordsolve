"use strict";

importScripts('./pkg/wordsolve.js');
const lib = wasm_bindgen;
const initLib = lib('./pkg/wordsolve_bg.wasm');

function js_log(message) {
    logAppend(message);
}

function js_now() {
    return performance.now();
}

function packWord(word) {
    if (!/^[a-z]{5}$/.test(word)) {
        throw new Error("invalid word");
    }
    let u = 0;
    for (let i = 0; i < word.length; i++) {
        u |= (word.charCodeAt(i) & 0b11111) << (5 * i);
    }
    return u;
}

function unpackWord(u) {
    const codes = [];
    for (let i = 0; i < 5; i++) {
        codes.push(0x60 + ((u >> (5 * i)) & 0b11111));
    }
    return String.fromCharCode(...codes);
}

function extractWords(words) {
    const queries = words.map(w => w.replace("!", "")).sort().map(packWord);
    const candidates = words
          .filter(w => !w.startsWith("!"))
          .sort()
          .map(packWord);
    return {candidates, queries};
}

function packResponse(response) {
    if (!/^[0-2]{5}$/.test(response)) {
        throw new Error("invalid response");
    }
    let u = 0;
    for (const [i, c] of [...response].entries()) {
        u |= ((c == "2") << (i + 5)) | ((c == "1") << i);
    }
    return u;
}

function unpackResponse(packed) {
    let chars = [];
    for (let i = 0; i < 5; i++) {
        chars.push(
            (packed >> (i + 5)) & 1 ? "2" :
                (packed >> i) & 1 ? "1" :
                "0");
    }
    return chars.join("");
}

function computeResponse(query, candidate) {
    const xor = query ^ candidate;
    let posMatch = 0;
    for (let i = 0; i < 5; ++i) {
        const m = !((xor >> (5 * i)) & 0b11111);
        posMatch |= m << i;
        candidate |= (m * 0b11111) << (5 * i);
    }
    let charMatch = 0;
    for (let i = 0; i < 5; ++i) {
        if (!((posMatch >> i) & 1)) {
            for (let j = 0; j < 5; ++j) {
                if (((query >> (5 * i)) & 0b11111) ==
                    ((candidate >> (5 * j)) & 0b11111)) {
                    charMatch |= 1 << i;
                    candidate |= 0b11111 << (5 * j);
                    break;
                }
            }
        }
    }
    return (posMatch << 5) | charMatch;
}

function log(message) {
    postMessage({cmd: "log", message});
}

function logAppend(message) {
    postMessage({cmd: "logAppend", message});
}

let preprocessed = null;
let preprocessedKey = null;

function preprocess(words) {
    let key = words.join(",");
    if (preprocessedKey == key) {
        return preprocessed;
    }
    console.time('preprocess');
    const {candidates, queries} = extractWords(words);
    let tick = 0;
    const nq = queries.length;
    const nc = candidates.length;
    const matrix = new Uint16Array(nq * nc);
    for (let q = 0; q < nq; ++q) {
        const query = queries[q];
        if (tick % 200 == 0) {
            const progress = q / nq;
            log(`Preprocessing... ${(progress * 100).toFixed(0)}%`);
        }
        for (let c = 0; c < nc; ++c) {
            const candidate = candidates[c];
            matrix[q * nc + c] = computeResponse(query, candidate);
            ++tick;
        }
    }
    console.timeEnd('preprocess');
    preprocessed = {queries, candidates, matrix, cache: new Map()};
    preprocessedKey = key;
    return preprocessed;
}

function binCandidatesM(responses, candidates) {
    const nc = candidates.length;
    const outcomes = new Map();
    for (const candidate of candidates) {
        const response = responses[candidate];
        let subcandidates = outcomes.get(response);
        if (subcandidates == null) {
            subcandidates = [];
            outcomes.set(response, subcandidates);
        }
        subcandidates.push(candidate);
    }
    const outcomeLists = [];
    for (const [_, subcandidates] of outcomes) {
        outcomeLists.push(subcandidates);
    }
    return outcomeLists;
}

function binCandidatesB(responses, candidates) {
    const nc = candidates.length;
    const responseCandidates = new Uint32Array(candidates);
    for (let ic = 0; ic < nc; ++ic) {
        const c = responseCandidates[ic];
        responseCandidates[ic] = (responses[c] << 16) | c;
    }
    responseCandidates.sort();
    const outcomeLists = [];
    let startIndex = 0;
    for (let ic = 0; ic < nc; ++ic) {
        const rc = responseCandidates[ic];
        const c = rc & 0xffff;
        const r = rc >> 16;
        responseCandidates[ic] = c;
        if (ic + 1 == nc || r != (responseCandidates[ic + 1] >> 16)) {
            outcomeLists.push(responseCandidates.subarray(startIndex, ic + 1));
            startIndex = ic + 1;
        }
    }
    return outcomeLists;
}

function heuristicSolve(p, queries, candidates) {
    const ns = p.candidates.length;
    const nc = candidates.length;
    if (nc <= 1) {
        throw new Error("expected nc > 1");
    }
    const decisions = [];
    const effectiveQueries = [];
    for (const query of queries) {
        const responses = p.matrix.subarray(query * ns, (query + 1) * ns);
        const outcomeLists = binCandidatesB(responses, candidates);
        if (outcomeLists.length <= 1) {
            continue;
        }
        let kMax = 0;
        let entropy = 0;
        for (const subcandidates of outcomeLists) {
            const k = subcandidates.length;
            if (kMax < k) {
                kMax = k;
            }
            const p = k / nc;
            entropy += -p * Math.log(p);
        }
        const kAvg = Math.exp(-entropy) * nc;
        effectiveQueries.push(query);
        decisions.push({query, outcomeLists, kMax, kAvg});
    }
    decisions.sort((x, y) => comparePair(x.kMax, x.kAvg, y.kMax, y.kAvg));
    return {decisions, effectiveQueries};
}

function comparePair(lx, ly, rx, ry) {
    return lx != rx ? lx - rx : ly - ry;
}

function solveWith(p, queries, candidates) {
    const key = candidates.join(",");
    if (p.counter % 100 == 0) {
        console.log("heartbeat");
    }
    ++p.counter;
    let result = p.cache.get(key);
    if (result == null) {
        const nc = candidates.length;
        if (nc <= 1) {
            return {avg: 0, max: 0};
        }
        const {decisions, effectiveQueries} = heuristicSolve(p, queries, candidates);
        const output = [];
        let dMin = {avg: Infinity, max: Infinity};
        for (const [i, decision] of decisions.entries()) {
            if (i >= 1) {
                break;
            }
            const d = solveWithDecision(p, effectiveQueries, decision);
            if (comparePair(dMin.max, dMin.avg, d.max, d.avg) > 0) {
                dMin = d;
            }
        }
        result = dMin;
        p.cache.set(key, result);
    }
    return result;
}

function solveWithDecision(p, queries, decision) {
    let dTotal = {avg: 0, max: 0};
    for (const candidates of decision.outcomeLists) {
        const d = solveWith(p, queries, candidates);
        dTotal.avg += d.avg;
        if (dTotal.max < d.max) {
            dTotal.max = d.max;
        }
    }
    if (!decision.outcomeLists.length) {
        console.error(decision.outcomeLists);
        throw ";";
    }
    dTotal.avg /= decision.outcomeLists.length;
    ++dTotal.max;
    ++dTotal.avg;
    return dTotal;
}

function solve(p, queries, candidates) {
    if (candidates.length >= (1 << 16)) {
        throw new Error("cannot have more than 65535 candidates");
    }
    const {decisions, effectiveQueries} = heuristicSolve(p, queries, candidates);
    const output = [];
    for (const [i, decision] of decisions.entries()) {
        if (i >= 1) {
            break;
        }
        console.time('solve-root');
        const d = solveWithDecision(p, effectiveQueries, decision);
        console.timeEnd('solve-root');
        postMessage({
            cmd: "appendQuery",
            query: [
                unpackWord(p.queries[decision.query]),
                decision.kMax,
                decision.kAvg,
                d.max,
                d.avg,
            ],
        });
    }
    return output.join("\n");
}

function range(n) {
    return new Array(n).fill(0).map((_, i) => i);
}

function applyPriorGuesses(p, guesses, candidates) {
    const packedGuesses = guesses.map(guess => ({
        query: packWord(guess.query),
        response: packResponse(guess.response),
    }));
    return candidates.filter(index => {
        const candidate = p.candidates[index];
        for (const {query, response} of packedGuesses) {
            if (computeResponse(query, candidate) != response) {
                return false;
            }
        }
        return true;
    });
}

function dumpCandidates(p, candidates) {
    const words = candidates.map(index => unpackWord(p.candidates[index]));
    words.sort();
    postMessage({
        cmd: "setCandidates",
        candidates: words,
    });
}

function runSolve(data) {
    log("Preprocessing...");
    console.time("preprocess");
//    const preprocessed = preprocess(data.words);
    const preprocessed = lib.preprocess(data.words.join(","));
    console.timeEnd("preprocess");
    return;
    const candidates = range(preprocessed.candidates.length);
    log("Filtering...");
    const filteredCandidates = applyPriorGuesses(
        preprocessed,
        data.guesses,
        candidates,
    );
    preprocessed.counter = 0;
    dumpCandidates(preprocessed, filteredCandidates);
    log("Solving...");
    solve(
        preprocessed,
        range(preprocessed.queries.length),
        filteredCandidates,
    );
}

async function main() {
    log("Initializing...");
    await initLib;
    log("Click 'Find strategy' to begin.");
    onmessage = e => ({
        runSolve,
    })[e.data.cmd](e.data);
}

main();
