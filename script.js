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

function loadWordsFromCache(storageKey, statusBar, wordList) {
    const list = localStorage.getItem(storageKey);
    if (!list) {
        statusBar.update("Please upload a word list first.");
        return;
    }
    const words = parseWords(list);
    statusBar.update(`Loaded ${words.length} word(s) from cache.`);
    wordList.words = words;
}

async function loadWordsFromFile(storageKey, input, statusBar, wordList) {
    statusBar.update("Loading...", -1);
    const [file] = input.files;
    if (!file) {
        statusBar.update("Error: Please select a file.");
        return;
    }
    try {
        const words = parseWords(await file.text());
        localStorage.setItem(storageKey, words.join("\n"));
        statusBar.update(`Loaded ${words.length} word(s).`);
        wordList.words = words;
    } catch (e) {
        statusBar.update(`Error: ${e}`);
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

function loadStatusBar() {
    const progressBar = document.getElementById("progress-bar");
    const statusMessage = document.getElementById("status-message");
    return {
        update(message, progress) {
            statusMessage.innerText = message || "\xa0";
            statusMessage.className = message.startsWith("Error: ") ? "error" : "";
            let progressText, progressOpacity = 1;
            if (progress != null && progress >= 0 && progress <= 1) {
                progressBar.value = progress;
                progressText = progress.toFixed(2) + "%";
            } else if (progress != null && progress < 0) {
                progressBar.removeAttribute("value");
                progressText = "";
            } else {
                progressOpacity = 0.5;
                progressBar.value = 0;
                progressText = "";
            }
            progressBar.style.opacity = progressOpacity;
            progressBar.innerText = progressText;
            progressBar.title = progressText;
        },
    };
}

function loadWordList(runner, statusBar) {
    const STORAGE_KEY = "wordsolve-wordlist";
    const wordList = {words: []};
    const button = document.getElementById("upload-word-list-button");
    const input = document.getElementById("word-list-input");
    loadWordsFromCache(STORAGE_KEY, statusBar, wordList);
    button.addEventListener("click", () => {
        loadWordsFromFile(STORAGE_KEY, input, statusBar, wordList);
    });
    return wordList;
}

function loadSolver(runner, statusBar, wordList) {
    const button = document.getElementById("solve-button");
    const textarea = document.getElementById("guesses-textarea");
    const queriesTable = document.getElementById("queries");
    const candidatesTable = document.getElementById("candidates");
    const maxBranching = document.getElementById("max-branching");
    runner.commands["UpdateStatus"] = ({message, progress}) => {
        statusBar.update(message, progress);
        if (progress == null) {
            button.disabled = false;
        }
    };
    runner.commands["ClearQueries"] = ({}) => {
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
        button.disabled = true;
        clearTable(queriesTable);
        if (!/^\d+$/.test(maxBranching.value)) {
            runner.commands["UpdateStatus"]({message: "Error: Max branching factor must be an integer."});
            return;
        }
        runner.worker.postMessage({
            "cmd": "RunSolve",
            "words": wordList.words,
            "guesses": textarea.value,
            "max_branching": +maxBranching.value,
        });
    });
}

function main() {
    try {
        const statusBar = loadStatusBar();
        const runner = loadRunner();
        const wordList = loadWordList(runner, statusBar);
        loadSolver(runner, statusBar, wordList);
    } catch (e) {
        document.body.className = "error";
        document.body.innerText = `Error: ${e}`;
        return null;
    }
}

main();
