"use strict";

function parseWords(lines) {
    const words = [];
    for (const line of lines.split("\n")) {
        const word = line.trim().toLowerCase();
        if (/^!?[a-z]{5}$/.test(word)) {
            words.push(word);
        }
    }
    return {words: words.join("\n"), length: words.length};
}

function loadWordsFromCache(storageKey, statusBar, wordList) {
    const cachedWords = localStorage.getItem(storageKey);
    if (!cachedWords) {
        statusBar.update("Please upload a word list first.");
        return;
    }
    const {words, length} = parseWords(cachedWords);
    statusBar.update(`Loaded ${length} word(s) from cache.`);
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
        const {words, length} = parseWords(await file.text());
        localStorage.setItem(storageKey, words);
        statusBar.update(`Loaded ${length} word(s).`);
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

// Can append either a string or an iterable of elements.
function appendContents(element, contents) {
    if (typeof contents == "string") {
        element.appendChild(document.createTextNode(contents));
    } else {
        for (const child of contents) {
            element.appendChild(child);
        }
    }
}

function removeAllChildren(element) {
    for (let i = element.children.length - 1; i >= 0; --i) {
        const child = element.children[i];
        element.removeChild(element.children[i]);
    }
}

function appendTableRow(table, row) {
    const tr = document.createElement("tr");
    for (const cell of row) {
        const td = document.createElement("td");
        appendContents(td, cell);
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
    const wordList = {words: ""};
    const button = document.getElementById("upload-word-list-button");
    const input = document.getElementById("word-list-input");
    loadWordsFromCache(STORAGE_KEY, statusBar, wordList);
    button.addEventListener("click", () => {
        loadWordsFromFile(STORAGE_KEY, input, statusBar, wordList);
    });
    return wordList;
}

function createDetails(summaryChildren) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    appendContents(summary, summaryChildren);
    details.appendChild(summary);
    return details;
}

function renderDecisionTree(decision, prefix) {
    const details = createDetails((prefix || "") + decision.query);
    const div = document.createElement("div");
    details.appendChild(div);
    function onToggle() {
        if (details.open) {
            for (const outcome of decision.outcomes) {
                const prefix = `${outcome.response} - `;
                div.appendChild(
                    outcome.next_decision == null
                        ? createDetails(prefix + "(win)")
                        : renderDecisionTree(outcome.next_decision, prefix));
            }
        } else {
            removeAllChildren(div);
        }
        details.removeEventListener("toggle", onToggle);
    }
    details.addEventListener("toggle", onToggle);
    return details;
}

function loadSolver(runner, statusBar, wordList) {
    const button = document.getElementById("solve-button");
    const textarea = document.getElementById("guesses-textarea");
    const queriesTable = document.getElementById("queries");
    const candidatesTable = document.getElementById("candidates");
    runner.commands["UpdateStatus"] = ({message, progress}) => {
        statusBar.update(message, progress);
        if (progress == null) {
            button.disabled = false;
        }
    };
    runner.commands["ReportStrategy"] = ({depths, depth_avg, decision_tree}) => {
        console.log("ReportStrategy", depths, decision_tree);
        const row = [
            `${depths.length - 1}`,
            `${depth_avg.toFixed(6)}`,
            `[${depths.join(", ")}]`,
            [renderDecisionTree(decision_tree)],
        ];
        appendTableRow(queriesTable, row);
    };
    runner.commands["SetCandidates"] = ({candidates}) => {
        removeAllChildren(candidatesTable);
        const fragment = new DocumentFragment();
        for (const [i, candidate] of candidates.entries()) {
            appendTableRow(fragment, [candidate]);
        }
        candidatesTable.appendChild(fragment);
        document.getElementById("num-candidates").innerText = candidates.length;
    };
    button.addEventListener("click", () => {
        button.disabled = true;
        removeAllChildren(queriesTable);
        runner.worker.postMessage({
            "cmd": "RunSolve",
            "words": wordList.words,
            "guesses": textarea.value,
            "num_roots": +document.getElementById("num-roots").value,
            "dk_trunc": +document.getElementById("dk-trunc").value,
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
