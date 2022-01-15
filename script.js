"use strict";

function parseWords(lines) {
    const words = [];
    for (const line of lines.split("\n")) {
        const word = line.trim().toLowerCase();
        if (/^!?[a-z]{5}$/.test(word)) {
            words.push(word);
        }
    }
    return words;
}

function loadWordsFromCache(storageKey, log, wordList) {
    const list = localStorage.getItem(storageKey);
    if (list) {
        const words = parseWords(list);
        log.innerText = `Loaded ${words.length} word(s) from cache.`;
        wordList.words = words;
    }
}

async function loadWordsFromFile(storageKey, input, log, wordList) {
    log.innerText = "Loading...";
    const [file] = input.files;
    if (!file) {
        log.innerText = "Error: Please select a file.";
        return;
    }
    try {
        const words = parseWords(await file.text());
        localStorage.setItem(storageKey, words.join("\n"));
        log.innerText = `Loaded ${words.length} word(s).`;
        wordList.words = words;
    } catch (e) {
        log.innerText = `Error: ${e}`;
    }
}

function loadRunner() {
    const commands = {};
    const worker = new Worker('worker.js');
    worker.addEventListener("message", e => {
        commands[e.data.cmd](e.data);
    });
    return {commands, worker};
}

function loadWordList(runner) {
    const STORAGE_KEY = "wordsolve-wordlist";
    const wordList = {words: []};
    const button = document.getElementById("upload-word-list-button");
    const input = document.getElementById("word-list-input");
    const log = document.getElementById("word-list-log");
    loadWordsFromCache(STORAGE_KEY, log, wordList);
    button.addEventListener("click", () => {
        loadWordsFromFile(STORAGE_KEY, input, log, wordList);
    });
    return wordList;
}

function clearTable(table) {
    for (let i = table.children.length - 1; i >= 0; --i) {
        const child = table.children[i];
        table.removeChild(table.children[i]);
    }
}

function appendTableRow(table, row) {
    const tr = document.createElement("tr");
    for (const cell of row) {
        const td = document.createElement("td");
        td.innerText = cell;
        tr.appendChild(td);
    }
    table.appendChild(tr);
}

function parseGuesses(text) {
    const guesses = [];
    for (const line of text.split("\n")) {
        const cleanedLine = line.split("#")[0].trim().toLowerCase();
        if (!cleanedLine) {
            continue;
        }
        const match = /^([^\s]+)\s+([^\s]+)$/.exec(cleanedLine);
        if (match == null) {
            throw new Error(
                "each line must have two parts: <guessed_word> <color_response>"
            );
        }
        const query = match[1];
        if (!/[a-z]{5}/.test(query)) {
            throw new Error("word must be 5 letters");
        }
        const response = match[2];
        if (!/[0-2]{5}/.test(response)) {
            throw new Error("response must be 5 digits, each from 0 to 2");
        }
        guesses.push({query, response});
    }
    return guesses;
}

function loadSolver(runner, wordList) {
    const button = document.getElementById("solve-button");
    const textarea = document.getElementById("guesses-textarea");
    const log = document.getElementById("solve-log");
    const queriesTable = document.getElementById("queries");
    const candidatesTable = document.getElementById("candidates");
    runner.commands["ClearLog"] = data => {
        log.innerText = "";
    };
    runner.commands["Log"] = data => {
        const message = data.message;
        if (message.startsWith("\x1b[2K\r")) {
            log.innerText =
                log.innerText.slice(0, log.innerText.lastIndexOf("\n") + 1)
                + message.slice(5);
        } else {
            log.innerText += message + "\n";
        }
    };
    runner.commands["ClearQueries"] = data => {
        for (const i = queriesTable.length - 1; i > 0; --i) {
            const child = queriesTable.children[i];
            queriesTable.removeChild(queriesTable.children[i]);
        }
    };
    runner.commands["AppendQuery"] = data => {
        appendTableRow(queriesTable, data.query);
    };
    runner.commands["SetCandidates"] = data => {
        clearTable(candidatesTable);
        const fragment = new DocumentFragment();
        for (const [i, candidate] of data.candidates.entries()) {
            appendTableRow(fragment, [candidate]);
        }
        candidatesTable.appendChild(fragment);
    };
    button.addEventListener("click", () => {
        clearTable(queriesTable);
        log.innerText = "";
        runner.worker.postMessage({
            cmd: "RunSolve",
            words: wordList.words,
            guesses: textarea.value,
        });
    });
}

function main() {
    try {
        const runner = loadRunner();
        const wordList = loadWordList(runner);
        loadSolver(runner, wordList);
    } catch (e) {
        document.body.className = "error";
        document.body.innerText = `Error: ${e}`;
        return null;
    }
}

main();
