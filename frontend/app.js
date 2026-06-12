/* ── PKM Dashboard App ──────────────────────────────────────────────────── */

const API = "";  // same origin

// ── State ──────────────────────────────────────────────────────────────────

let documents = [];
let sessionId = sessionStorage.getItem("pkm_session") || crypto.randomUUID();
sessionStorage.setItem("pkm_session", sessionId);
let currentDomain = localStorage.getItem("pkm_domain") || "general";
let domains = [];
let questionDocuments = [];
let activeQuestionDocId = null;
let activeQuestion = null;
let lastQuestionResult = null;
const GRAPH_PALETTE = ["#e15759", "#4e79a7", "#59a14f", "#f28e2b", "#b07aa1",
                       "#76b7b2", "#edc948", "#ff9da7", "#9c755f", "#bab0ac"];
const GRAPH_CAT_COLORS = { "is-a": "#4e79a7", "part-of": "#59a14f", "uses": "#76b7b2",
                           "contrasts": "#e15759", "causes": "#f28e2b", "related": "#6b6b75" };
const graphState = {
    cy: null,
    payload: null,
    topK: 3,              // max co-occurrence links kept per concept
    layers: { triple: true, semantic: true, cooccur: false, trail: true },
    showBadges: true,     // show source-document badges under concept labels
    dateCut: null,        // null = now (no temporal filter)
    dateRange: null,      // [minMs, maxMs] from payload
    selectedDocs: null,   // Set of doc ids whose concepts are visible; null = all
    ego: null,
};

function graphDocFilterKey() {
    return `pkm_graph_docs_${currentDomain}`;
}

// ── Init ───────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    initDomains();
    initRouter();
    initUpload();
    initSearch();
    initChat();
    initChatWidget();
    initQuestions();
    initConnections();
    loadDocuments();
    loadStats();
});

async function apiFetch(path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("X-PKM-Domain", currentDomain);
    return fetch(`${API}${path}`, { ...options, headers });
}

function initDomains() {
    renderDomains();
    const createInput = document.getElementById("domain-create-input");

    document.getElementById("domain-select").addEventListener("change", async (event) => {
        currentDomain = event.target.value;
        localStorage.setItem("pkm_domain", currentDomain);
        resetDomainScopedState();
        await loadDomains();
        await loadDocuments();
        await loadStats();
        if ((window.location.hash.slice(1) || "documents") === "questions") {
            await loadQuestions();
        }
        if ((window.location.hash.slice(1) || "documents") === "connections") {
            await renderConnections();
        }
    });

    async function createDomainFromInput() {
        const name = createInput.value.trim();
        if (!name) return;
        try {
            const res = await apiFetch("/api/domains", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });
            const data = await res.json();
            if (!res.ok) {
                alert(data.error || "Failed to create subject.");
                return;
            }
            currentDomain = data.domain.id;
            localStorage.setItem("pkm_domain", currentDomain);
            createInput.value = "";
            resetDomainScopedState();
            await loadDomains();
            await loadDocuments();
            await loadStats();
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    }

    document.getElementById("domain-create-btn").addEventListener("click", createDomainFromInput);
    createInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            createDomainFromInput();
        }
    });

    loadDomains();
}

async function loadDomains() {
    try {
        const res = await apiFetch("/api/domains");
        const data = await res.json();
        domains = data.domains || [];
        if (!domains.find(domain => domain.id === currentDomain) && domains.length) {
            currentDomain = domains[0].id;
            localStorage.setItem("pkm_domain", currentDomain);
        }
        renderDomains();
    } catch (err) {
        console.error("Failed to load domains:", err);
    }
}

function renderDomains() {
    const select = document.getElementById("domain-select");
    const visibleDomains = domains.length
        ? domains
        : [{ id: currentDomain, name: formatDomainName(currentDomain) }];
    select.innerHTML = visibleDomains.map(domain =>
        `<option value="${escapeHtml(domain.id)}" ${domain.id === currentDomain ? "selected" : ""}>${escapeHtml(domain.name)}</option>`
    ).join("");
}

function formatDomainName(domainId) {
    return domainId
        .split("-")
        .filter(Boolean)
        .map(part => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ") || "General";
}

function resetDomainScopedState() {
    documents = [];
    questionDocuments = [];
    activeQuestionDocId = null;
    activeQuestion = null;
    lastQuestionResult = null;
    graphState.payload = null;
}

// ── Router ─────────────────────────────────────────────────────────────────

function initRouter() {
    const navItems = document.querySelectorAll(".nav-item");
    const pages = document.querySelectorAll(".page");

    function navigate(page) {
        navItems.forEach(n => n.classList.remove("active"));
        pages.forEach(p => p.classList.remove("active"));

        const navItem = document.querySelector(`.nav-item[data-page="${page}"]`);
        const pageEl = document.getElementById(`page-${page}`);

        if (navItem) navItem.classList.add("active");
        if (pageEl) pageEl.classList.add("active");
        updateChatWidgetVisibility(page);

        if (page === "connections") renderConnections();
        if (page === "summaries") renderSummaries();
        if (page === "questions") loadQuestions();
    }

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            window.location.hash = page;
            navigate(page);
        });
    });

    // Handle initial hash
    const hash = window.location.hash.slice(1) || "documents";
    navigate(hash);

    window.addEventListener("hashchange", () => {
        navigate(window.location.hash.slice(1) || "documents");
    });
}

// ── Upload ─────────────────────────────────────────────────────────────────

function initUpload() {
    // Tab switching
    const tabs = document.querySelectorAll(".upload-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            document.querySelectorAll(".upload-content").forEach(c => c.classList.add("hidden"));
            document.getElementById(`tab-${tab.dataset.tab}`).classList.remove("hidden");
        });
    });

    // PDF upload
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        const files = Array.from(e.dataTransfer.files).filter(f => f.name.toLowerCase().endsWith(".pdf"));
        if (files.length) uploadPDFs(files);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            uploadPDFs(Array.from(fileInput.files));
            fileInput.value = "";
        }
    });

    // URL
    document.getElementById("url-btn").addEventListener("click", addURL);
    document.getElementById("url-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") addURL();
    });

    // Text
    document.getElementById("text-btn").addEventListener("click", addText);
}

async function uploadPDFs(files) {
    for (const file of files) {
        showLoading(`Uploading ${file.name}...`);
        try {
            const form = new FormData();
            form.append("file", file);
            const res = await apiFetch("/api/documents/upload-pdf", { method: "POST", body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Upload failed");
        } catch (err) {
            alert(`Error uploading ${file.name}: ${err.message}`);
        }
    }
    hideLoading();
    loadDocuments();
    loadStats();
}

async function addURL() {
    const input = document.getElementById("url-input");
    const url = input.value.trim();
    if (!url) return;

    showLoading("Fetching and indexing URL...");
    try {
        const res = await apiFetch("/api/documents/add-url", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to add URL");
        input.value = "";
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
    hideLoading();
    loadDocuments();
    loadStats();
}

async function addText() {
    const titleInput = document.getElementById("text-title");
    const contentInput = document.getElementById("text-content");
    const title = titleInput.value.trim();
    const text = contentInput.value.trim();
    if (!title || !text) { alert("Please provide both a title and text content."); return; }

    showLoading("Indexing text...");
    try {
        const res = await apiFetch("/api/documents/add-text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, text }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to add text");
        titleInput.value = "";
        contentInput.value = "";
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
    hideLoading();
    loadDocuments();
    loadStats();
}

// ── Documents ──────────────────────────────────────────────────────────────

async function loadDocuments() {
    try {
        const res = await apiFetch("/api/documents");
        const data = await res.json();
        documents = data.documents || [];
        renderDocuments();
        if ((window.location.hash.slice(1) || "documents") === "connections") {
            renderConnections();
        }
        if ((window.location.hash.slice(1) || "documents") === "questions") {
            loadQuestions();
        }
    } catch (err) {
        console.error("Failed to load documents:", err);
    }
}

function renderDocuments() {
    const list = document.getElementById("doc-list");
    if (!documents.length) {
        list.innerHTML = '<p class="empty-state">No documents yet. Upload a PDF, add a URL, or paste some text to get started.</p>';
        return;
    }

    list.innerHTML = documents.map(doc => `
        <div class="doc-card" data-id="${doc.doc_id}">
            <div class="doc-info">
                <div class="doc-title">
                    <span class="doc-type ${doc.source_type}">${doc.source_type}</span>
                    ${escapeHtml(doc.title)}
                </div>
                <div class="doc-meta">
                    <span>${doc.chunk_count} chunks</span>
                    ${doc.added_at ? `<span>${new Date(doc.added_at).toLocaleDateString()}</span>` : ""}
                    ${doc.connection_count ? `<span>${doc.connection_count} connections</span>` : ""}
                </div>
            </div>
            <div class="doc-actions">
                <button class="btn btn-danger" onclick="deleteDocument('${doc.doc_id}')">Delete</button>
            </div>
        </div>
    `).join("");
}

async function deleteDocument(docId) {
    if (!confirm("Delete this document and all its data?")) return;

    try {
        const res = await apiFetch(`/api/documents/${docId}`, { method: "DELETE" });
        if (!res.ok) throw new Error("Delete failed");
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
    loadDocuments();
    loadStats();
}

// ── Stats ──────────────────────────────────────────────────────────────────

async function loadStats() {
    try {
        const res = await apiFetch("/api/stats");
        const stats = await res.json();
        document.getElementById("stat-docs").textContent = stats.total_documents || 0;
        document.getElementById("stat-chunks").textContent = stats.total_chunks || 0;
        document.getElementById("stat-qa").textContent = stats.total_questions || 0;
    } catch (err) {
        console.error("Failed to load stats:", err);
    }
}

// ── Search ─────────────────────────────────────────────────────────────────

function initSearch() {
    const input = document.getElementById("search-input");
    const btn = document.getElementById("search-btn");
    const results = document.getElementById("search-results");

    let debounceTimer;

    btn.addEventListener("click", () => doSearch(input.value));
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") doSearch(input.value);
    });

    input.addEventListener("input", () => {
        clearTimeout(debounceTimer);
        if (!input.value.trim()) {
            results.classList.add("hidden");
            return;
        }
        debounceTimer = setTimeout(() => doSearch(input.value), 500);
    });

    document.addEventListener("click", (e) => {
        if (!results.contains(e.target) && e.target !== input) {
            results.classList.add("hidden");
        }
    });
}

async function doSearch(query) {
    query = query.trim();
    const results = document.getElementById("search-results");
    if (!query) { results.classList.add("hidden"); return; }

    try {
        const res = await apiFetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, top_k: 8 }),
        });
        const data = await res.json();

        if (!data.results || !data.results.length) {
            results.innerHTML = '<div class="empty-state" style="padding:16px">No results found.</div>';
        } else {
            results.innerHTML = data.results.map(r => `
                <div class="search-result-item">
                    <div class="search-result-title">${escapeHtml(r.title)}</div>
                    <div class="search-result-text">${escapeHtml(r.chunk_text.substring(0, 200))}...</div>
                    <div class="search-result-score">Score: ${r.score} | ${r.source || "text note"}</div>
                </div>
            `).join("");
        }
        results.classList.remove("hidden");
    } catch (err) {
        console.error("Search error:", err);
    }
}

// ── Chat ───────────────────────────────────────────────────────────────────

function initChat() {
    const input = document.getElementById("chat-input");
    const btn = document.getElementById("chat-send");

    btn.addEventListener("click", () => sendChat({
        inputId: "chat-input",
        messagesId: "chat-messages",
    }));
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChat({
            inputId: "chat-input",
            messagesId: "chat-messages",
        });
    });
}

function initChatWidget() {
    const input = document.getElementById("chat-widget-input");
    const btn = document.getElementById("chat-widget-send");
    const toggle = document.getElementById("chat-widget-toggle");
    const widget = document.getElementById("chat-widget");

    btn.addEventListener("click", () => sendChat({
        inputId: "chat-widget-input",
        messagesId: "chat-widget-messages",
    }));
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChat({
            inputId: "chat-widget-input",
            messagesId: "chat-widget-messages",
        });
    });
    toggle.addEventListener("click", () => {
        widget.classList.toggle("open");
        toggle.textContent = widget.classList.contains("open") ? "-" : "QC";
    });

    updateChatWidgetVisibility(window.location.hash.slice(1) || "documents");
}

function updateChatWidgetVisibility(page) {
    const widget = document.getElementById("chat-widget");
    if (!widget) return;
    widget.classList.toggle("hidden", page === "chat" || page === "connections");
}

async function sendChat({ inputId, messagesId }) {
    const input = document.getElementById(inputId);
    const message = input.value.trim();
    if (!message) return;

    input.value = "";
    appendChatMsg(messagesId, "user", message);

    // Show typing indicator
    const typingEl = appendChatMsg(messagesId, "assistant", "Thinking...");
    typingEl.style.opacity = "0.5";

    try {
        const res = await apiFetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId }),
        });
        const data = await res.json();

        typingEl.remove();
        appendChatAnswer(messagesId, data);
        loadStats();
    } catch (err) {
        typingEl.remove();
        appendChatMsg(messagesId, "assistant", `Error: ${err.message}`);
    }
}

function appendChatMsg(messagesId, role, text) {
    const container = document.getElementById(messagesId);
    // Remove welcome message
    const welcome = container.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const div = document.createElement("div");
    div.className = `chat-msg ${role}`;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

function appendChatAnswer(messagesId, data) {
    const container = document.getElementById(messagesId);

    const div = document.createElement("div");
    div.className = "chat-msg assistant";

    let html = "";
    if (data.confidence) {
        html += `<span class="chat-confidence ${data.confidence}">${data.confidence}</span><br>`;
    }
    html += escapeHtml(data.answer);

    if (data.sources && data.sources.length) {
        html += '<div class="chat-sources">Sources: ';
        html += data.sources.map(s => escapeHtml(s.title)).join(", ");
        html += "</div>";
    }

    if (data.related_docs && data.related_docs.length) {
        html += '<div class="chat-related-docs"><div class="chat-related-title">Also found in:</div>';
        html += data.related_docs.map(doc =>
            `<div class="chat-related-item"><strong>${escapeHtml(doc.doc)}</strong> <span>(${escapeHtml(doc.reason)})</span></div>`
        ).join("");
        html += "</div>";
    }

    if (data.connections && data.connections.length) {
        html += '<div class="chat-graph-connections"><div class="chat-related-title">Concept links:</div>';
        html += data.connections.map(link => {
            const concepts = Array.isArray(link.concept) ? link.concept.map(escapeHtml).join(", ") : "";
            return `<div class="chat-related-item"><strong>${escapeHtml(link.from_title)}</strong> → <strong>${escapeHtml(link.to_title)}</strong> <span>(${concepts})</span></div>`;
        }).join("");
        html += "</div>";
    }

    div.innerHTML = html;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// ── Summaries ──────────────────────────────────────────────────────────────

function renderSummaries() {
    const list = document.getElementById("summaries-list");
    const docsWithSummaries = documents.filter(d => d.summary);

    if (!docsWithSummaries.length) {
        list.innerHTML = '<p class="empty-state">No summaries yet. Upload documents to generate summaries automatically.</p>';
        return;
    }

    list.innerHTML = docsWithSummaries.map(doc => `
        <div class="summary-card">
            <div class="summary-header" onclick="toggleSummary(this)">
                <span class="summary-title">
                    <span class="doc-type ${doc.source_type}">${doc.source_type}</span>
                    ${escapeHtml(doc.title)}
                </span>
                <span class="summary-toggle">&#9660;</span>
            </div>
            <div class="summary-body">${escapeHtml(doc.summary)}</div>
        </div>
    `).join("");
}

// ── Questions ─────────────────────────────────────────────────────────────

function initQuestions() {
    document.getElementById("questions-refresh-btn").addEventListener("click", loadQuestions);
    document.getElementById("generate-doc-questions-btn").addEventListener("click", generateQuestionsForActiveDoc);
    document.getElementById("adaptive-next-btn").addEventListener("click", loadAdaptiveQuestion);
}

async function loadQuestions() {
    try {
        const res = await apiFetch("/api/questions");
        const data = await res.json();
        questionDocuments = data.documents || [];
        renderQuestionDocList();

        if (!activeQuestionDocId && questionDocuments.length) {
            activeQuestionDocId = questionDocuments[0].doc_id;
        }
        if (activeQuestionDocId) {
            const currentDoc = questionDocuments.find(doc => doc.doc_id === activeQuestionDocId);
            if (currentDoc) {
                activeQuestion = currentDoc.questions?.[0] || null;
            }
        }
        renderQuestionsPanel();
        await loadAnswerHistory();
    } catch (err) {
        console.error("Failed to load questions:", err);
    }
}

function renderQuestionDocList() {
    const list = document.getElementById("questions-doc-list");
    if (!questionDocuments.length) {
        list.innerHTML = '<p class="empty-state">Upload PDFs first to generate questions.</p>';
        return;
    }

    list.innerHTML = questionDocuments.map(doc => `
        <button class="questions-doc-item ${doc.doc_id === activeQuestionDocId ? "active" : ""}" onclick="selectQuestionDocument('${doc.doc_id}')">
            <div class="questions-doc-item-title">${escapeHtml(doc.title)}</div>
            <div class="questions-doc-item-meta">${doc.question_count} questions · ${escapeHtml((doc.concepts || []).slice(0, 2).join(", ") || "No concepts yet")}</div>
        </button>
    `).join("");
}

function selectQuestionDocument(docId) {
    activeQuestionDocId = docId;
    lastQuestionResult = null;
    const currentDoc = questionDocuments.find(doc => doc.doc_id === docId);
    activeQuestion = currentDoc?.questions?.[0] || null;
    renderQuestionDocList();
    renderQuestionsPanel();
}

function renderQuestionsPanel() {
    const activeLabel = document.getElementById("questions-active-doc");
    const wrap = document.getElementById("questions-card-wrap");
    const currentDoc = questionDocuments.find(doc => doc.doc_id === activeQuestionDocId);

    if (!currentDoc) {
        activeLabel.textContent = "Pick a PDF to begin.";
        wrap.innerHTML = '<div class="empty-state">Choose a PDF from the left to study it.</div>';
        return;
    }

    activeLabel.textContent = `${currentDoc.title} · ${currentDoc.question_count} questions`;

    if (!currentDoc.question_count) {
        wrap.innerHTML = '<div class="empty-state">No questions generated for this PDF yet. Click "Generate Questions".</div>';
        return;
    }

    const question = activeQuestion || currentDoc.questions[0];
    wrap.innerHTML = renderQuestionCard(question, currentDoc.title, lastQuestionResult);
}

function renderQuestionCard(question, docTitle, result) {
    if (question.type === "short_answer") {
        return renderShortAnswerCard(question, docTitle, result);
    }

    return renderMultipleChoiceCard(question, docTitle, result);
}

function renderMultipleChoiceCard(question, docTitle, result) {
    const selectedIndex = result?.selected_index;
    const correctIndex = result?.answer_index;
    const feedbackHtml = result ? `
        <div class="question-feedback">
            <div class="question-feedback-status">${result.correct ? "Correct" : "Not quite"}</div>
            <div class="question-explanation">${escapeHtml(result.explanation || "")}</div>
            <div class="question-mastery">Topic mastery: ${Math.round((result.mastery || 0) * 100)}%</div>
            <div class="question-next-wrap">
                <button class="btn btn-secondary" onclick="loadAdaptiveQuestion()">Next Adaptive Question</button>
            </div>
        </div>
    ` : "";

    return `
        <div class="question-card">
            <div class="question-card-header">
                <span class="question-topic-badge">${escapeHtml(question.topic || "Core concept")}</span>
                <span class="question-difficulty">${escapeHtml(question.difficulty || "medium")}</span>
            </div>
            <div class="question-meta">${escapeHtml(docTitle)}</div>
            <div class="question-prompt">${escapeHtml(question.prompt)}</div>
            <div class="question-options">
                ${(question.options || []).map((option, index) => {
                    let stateClass = "";
                    if (result) {
                        if (index === correctIndex) stateClass = "correct";
                        else if (index === selectedIndex) stateClass = "wrong";
                    }
                    return `<button class="question-option ${stateClass}" onclick="submitQuestionAnswer('${question.doc_id || activeQuestionDocId}', '${question.id}', ${index})" ${result ? "disabled" : ""}>${escapeHtml(option)}</button>`;
                }).join("")}
            </div>
            ${feedbackHtml}
        </div>
    `;
}

function renderShortAnswerCard(question, docTitle, result) {
    const feedbackHtml = result ? `
        <div class="question-feedback">
            <div class="question-feedback-status">
                Score: ${Math.round(result.score || 0)}%
            </div>

            <div class="question-mastery">
                Topic mastery: ${Math.round((result.mastery || 0) * 100)}%
            </div>

            <div class="question-explanation">
                <strong>AI Feedback:</strong><br>
                ${escapeHtml(result.feedback || "")}
            </div>

            <div class="question-explanation">
                <strong>Sample Answer:</strong><br>
                ${escapeHtml(result.sample_answer || question.sample_answer || "")}
            </div>

            <div class="question-next-wrap">
                <button class="btn btn-secondary" onclick="loadAdaptiveQuestion()">
                    Next Adaptive Question
                </button>
            </div>
        </div>
    ` : "";

    return `
        <div class="question-card">
            <div class="question-card-header">
                <span class="question-topic-badge">${escapeHtml(question.topic || "Core concept")}</span>
                <span class="question-difficulty">${escapeHtml(question.difficulty || "medium")}</span>
            </div>
            <div class="question-meta">${escapeHtml(docTitle)}</div>
            <div class="question-prompt">${escapeHtml(question.prompt)}</div>

            <textarea id="short-answer-input" class="short-answer-input" placeholder="Write your answer here..." ${result ? "disabled" : ""}></textarea>

            <button class="btn btn-primary" onclick="submitShortAnswer('${question.doc_id || activeQuestionDocId}', '${question.id}')" ${result ? "disabled" : ""}>
                Submit Answer
            </button>

            ${feedbackHtml}
        </div>
    `;
}

async function submitShortAnswer(docId, questionId) {
    const answer = document.getElementById("short-answer-input").value;

    if (!answer.trim()) {
        alert("Please enter an answer.");
        return;
    }

    showLoading("Evaluating answer...");

    try {
        const res = await fetch(`${API}/api/questions/short-answer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: "default",
                doc_id: docId,
                question_id: questionId,
                answer_text: answer,
            }),
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || "Failed to evaluate answer");
        }

        lastQuestionResult = data.result;

        renderQuestionsPanel();
        await loadAnswerHistory();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }

    hideLoading();
}

async function generateQuestionsForActiveDoc() {
    if (!activeQuestionDocId) return;

    const questionType = document.getElementById("question-type-select").value;

    showLoading("Generating study questions...");
    try {
        const res = await apiFetch("/api/questions/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                doc_id: activeQuestionDocId,
                question_type: questionType,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to generate questions");
        activeQuestion = data.questions?.[0] || null;
        lastQuestionResult = null;
        await loadQuestions();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
    hideLoading();
}

async function submitQuestionAnswer(docId, questionId, selectedIndex) {
    try {
        const res = await apiFetch("/api/questions/answer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: "default",
                doc_id: docId,
                question_id: questionId,
                selected_index: selectedIndex,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to submit answer");
        lastQuestionResult = { ...data.result, selected_index: selectedIndex };
        if (activeQuestion) activeQuestion.doc_id = docId;
        renderQuestionsPanel();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

async function loadAdaptiveQuestion() {
    try {
        const res = await apiFetch("/api/questions/next", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, doc_id: activeQuestionDocId }),
        });
        const data = await res.json();
        activeQuestion = data.question;
        lastQuestionResult = null;
        if (data.question?.doc_id) {
            activeQuestionDocId = data.question.doc_id;
        }
        renderQuestionDocList();
        renderQuestionsPanel();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

function toggleSummary(header) {
    const body = header.nextElementSibling;
    const toggle = header.querySelector(".summary-toggle");
    body.classList.toggle("open");
    toggle.classList.toggle("open");
}

// ── Connections / Knowledge Map ────────────────────────────────────────────

function initConnections() {
    document.getElementById("refresh-connections-btn").addEventListener("click", async () => {
        showLoading("Computing knowledge connections...");
        try {
            const res = await apiFetch("/api/connections/refresh", { method: "POST" });
            if (!res.ok) throw new Error("Refresh failed");
            await loadDocuments();
            await renderConnections();
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
        hideLoading();
    });

    document.getElementById("graph-topk").addEventListener("input", (e) => {
        graphState.topK = parseInt(e.target.value, 10);
        document.getElementById("graph-topk-value").textContent = graphState.topK;
        rebuildGraphView();
    });

    let dateDebounce = null;
    document.getElementById("graph-date").addEventListener("input", (e) => {
        const pct = parseInt(e.target.value, 10);
        const valueEl = document.getElementById("graph-date-value");
        if (pct >= 100 || !graphState.dateRange) {
            graphState.dateCut = null;
            valueEl.textContent = "now";
        } else {
            const [min, max] = graphState.dateRange;
            const cut = new Date(min + (max - min) * pct / 100);
            graphState.dateCut = cut.toISOString();
            valueEl.textContent = graphState.dateCut.slice(0, 10);
        }
        clearTimeout(dateDebounce);
        dateDebounce = setTimeout(rebuildGraphView, 150);
    });

    const layerButtons = {
        "graph-show-relations": "triple",
        "graph-show-semantic": "semantic",
        "graph-show-cooccur": "cooccur",
        "graph-show-trails": "trail",
    };
    Object.entries(layerButtons).forEach(([btnId, layer]) => {
        document.getElementById(btnId).addEventListener("click", (e) => {
            e.target.classList.toggle("active");
            graphState.layers[layer] = e.target.classList.contains("active");
            rebuildGraphView();
        });
    });

    document.getElementById("graph-show-badges").addEventListener("click", (e) => {
        e.target.classList.toggle("active");
        graphState.showBadges = e.target.classList.contains("active");
        rebuildGraphView();
    });

    document.getElementById("graph-fit").addEventListener("click", () => {
        if (graphState.cy) graphState.cy.fit(undefined, 40);
    });

    document.getElementById("graph-search").addEventListener("input", (e) => {
        applyGraphSearch(e.target.value.trim().toLowerCase());
    });
    document.getElementById("graph-search").addEventListener("keydown", (e) => {
        if (e.key !== "Enter" || !graphState.cy) return;
        const hit = graphState.cy.nodes(".search-hit").first();
        if (hit.length) graphState.cy.animate({ center: { eles: hit }, zoom: 1.6, duration: 300 });
    });

    document.getElementById("graph-docs-all").addEventListener("click", () => setGraphDocFilter(null));
    document.getElementById("graph-docs-none").addEventListener("click", () => setGraphDocFilter(new Set()));

    document.getElementById("graph-ego-banner").addEventListener("click", exitGraphEgo);
}

async function renderConnections() {
    const emptyEl = document.getElementById("connections-empty");
    const wrapEl = document.getElementById("cy-wrap");

    let payload = { nodes: [], edges: [], communities: [], gaps: [], documents: [] };
    try {
        const res = await apiFetch("/api/graph");
        if (res.ok) payload = await res.json();
    } catch (err) {
        console.error("Failed to load graph:", err);
    }
    graphState.payload = payload;

    // temporal range from payload timestamps
    const stamps = payload.nodes.map(n => Date.parse(n.created_at)).filter(t => !isNaN(t));
    graphState.dateRange = stamps.length ? [Math.min(...stamps), Math.max(...stamps)] : null;

    // restore per-domain document filter
    try {
        const saved = localStorage.getItem(graphDocFilterKey());
        graphState.selectedDocs = saved ? new Set(JSON.parse(saved)) : null;
    } catch { graphState.selectedDocs = null; }

    const hasConcepts = payload.nodes.length > 0;
    wrapEl.style.display = hasConcepts ? "block" : "none";
    emptyEl.style.display = hasConcepts ? "none" : "block";

    renderGraphDocFilter(payload.documents || []);
    renderCommunityPanel(payload);
    renderGapPanel(payload);
    if (hasConcepts) rebuildGraphView();
}

// Short badge for a document title: leading chapter number ("3.Software..." -> "3")
// or the first word as fallback.
function docBadge(title) {
    const m = /^(\d+)\./.exec(title || "");
    if (m) return m[1];
    return (title || "?").split(/\s+/)[0].slice(0, 6);
}

function graphCyElements(payload) {
    const { topK, layers, dateCut, selectedDocs, showBadges } = graphState;
    const inDate = (iso) => !dateCut || !iso || isNaN(Date.parse(iso)) || iso <= dateCut;

    // concept nodes filtered by document selection + temporal cutoff
    const visible = payload.nodes.filter(n =>
        inDate(n.created_at) &&
        (selectedDocs === null || (n.doc_ids || []).some(id => selectedDocs.has(id))));
    const ids = new Set(visible.map(n => n.id));

    const nodes = visible.map(n => {
        const badge = (n.docs || []).map(docBadge).join("·");
        const display = showBadges && badge ? `${n.label}\n[${badge}]` : n.label;
        return { data: { ...n, display } };
    });

    // candidate edges among visible nodes, per active layers
    const candidates = payload.edges.filter(e =>
        ids.has(e.source) && ids.has(e.target) && layers[e.kind] !== false);

    // dense kinds (cooccur/semantic) keep only each node's top-K strongest
    // so within-document cliques never turn the map into a hairball
    const pruneTopK = (kind) => {
        const pool = candidates.filter(e => e.kind === kind).sort((a, b) => b.weight - a.weight);
        const perNode = new Map();
        const kept = [];
        for (const e of pool) {
            const cs = perNode.get(e.source) || 0;
            const ct = perNode.get(e.target) || 0;
            if (cs < topK || ct < topK) {
                kept.push(e);
                perNode.set(e.source, cs + 1);
                perNode.set(e.target, ct + 1);
            }
        }
        return kept;
    };
    const denseKinds = ["cooccur", "semantic"];
    const kept = denseKinds.flatMap(pruneTopK);
    const others = candidates.filter(e => !denseKinds.includes(e.kind));

    const normFor = {};
    denseKinds.forEach(kind => {
        const ws = candidates.filter(e => e.kind === kind).map(e => e.weight);
        normFor[kind] = [Math.min(...ws, 1), Math.max(...ws, 1.001)];
    });
    const edges = [...kept, ...others].map(e => {
        const [wMin, wMax] = normFor[e.kind] || [0, 1];
        return {
            data: {
                ...e,
                norm: (e.weight - wMin) / (wMax - wMin),
                label: e.kind === "cooccur"
                    ? (e.shared_docs || []).join(" · ")
                    : (e.label || e.kind),
            },
        };
    });
    // drop nodes that end up isolated after layer filtering and top-K pruning
    const connected = new Set();
    edges.forEach(e => { connected.add(e.data.source); connected.add(e.data.target); });
    const visibleNodes = nodes.filter(n => connected.has(n.data.id));
    return [...visibleNodes, ...edges];
}

function rebuildGraphView() {
    const payload = graphState.payload;
    if (!payload) return;
    graphState.ego = null;
    document.getElementById("graph-ego-banner").classList.remove("show");

    if (graphState.cy) graphState.cy.destroy();
    graphState.cy = cytoscape({
        container: document.getElementById("cy"),
        elements: graphCyElements(payload),
        style: [
            { selector: "node", style: {
                "background-color": (ele) => GRAPH_PALETTE[Math.max(ele.data("community"), 0) % GRAPH_PALETTE.length],
                "width": (ele) => 14 + ele.data("freq") * 6 + ele.data("centrality") * 40,
                "height": (ele) => 14 + ele.data("freq") * 6 + ele.data("centrality") * 40,
                "label": "data(display)", "color": "#d8dbe4", "font-size": 10,
                "text-valign": "bottom", "text-margin-y": 6,
                "text-max-width": "110px", "text-wrap": "wrap",
                "border-width": 2, "border-color": "#0c0e13",
            }},
            { selector: "node[kind='entity']", style: {
                "shape": "diamond", "border-color": "#4a5a7a",
            }},
            { selector: "edge", style: { "curve-style": "haystack", "haystack-radius": 0.3,
                "line-color": "#3a3f50", "width": 1, "opacity": 0.75 }},
            { selector: "edge[kind='cooccur']", style: {
                "line-style": "dashed", "line-color": "#3d4252",
                "width": (ele) => 1 + ele.data("norm") * 2 }},
            { selector: "edge[kind='semantic']", style: {
                "line-color": "#8a93b8",
                "width": (ele) => 1.2 + ele.data("norm") * 3 }},
            { selector: "edge[kind='triple']", style: {
                "line-color": (ele) => GRAPH_CAT_COLORS[ele.data("category")] || "#6b6b75", "width": 2,
                "target-arrow-shape": "triangle", "arrow-scale": 0.8, "curve-style": "bezier",
                "target-arrow-color": (ele) => GRAPH_CAT_COLORS[ele.data("category")] || "#6b6b75" }},
            { selector: "edge[kind='trail']", style: { "line-style": "dotted", "line-color": "#4e79a7", "width": 2.5, "curve-style": "bezier" }},
            { selector: ".label-hidden", style: { "text-opacity": 0 }},
            { selector: ".label-hidden:selected, .label-hidden:active", style: { "text-opacity": 1 }},
            { selector: ".faded", style: { "opacity": 0.08, "text-opacity": 0.03 }},
            { selector: ".hl-edge", style: {
                "label": "data(label)", "font-size": 8.5, "color": "#e8b34b",
                "text-background-color": "#0c0e13", "text-background-opacity": 0.85, "text-background-padding": "3px",
                "text-wrap": "ellipsis", "text-max-width": "180px", "curve-style": "bezier" }},
            { selector: "node.search-hit", style: { "border-color": "#e8b34b", "border-width": 3 }},
        ],
        layout: {
            name: "cose", animate: false, nodeRepulsion: 250000, idealEdgeLength: 110,
            gravity: 0.35, padding: 30, componentSpacing: 120, nodeOverlap: 30,
        },
        wheelSensitivity: 0.2,
    });
    wireGraphInteractions();
    wireZoomAdaptiveLabels();
    renderGraphLegend(graphState.payload);
    applyGraphSearch(document.getElementById("graph-search").value.trim().toLowerCase());
}

// Overview readability: below this zoom, only hub concepts keep labels.
const LABEL_ZOOM_THRESHOLD = 0.9;

function wireZoomAdaptiveLabels() {
    const cy = graphState.cy;
    const minor = cy.nodes().filter(n => n.data("freq") < 2 && n.data("centrality") < 0.05);
    let scheduled = false;
    const update = () => {
        scheduled = false;
        minor.toggleClass("label-hidden", cy.zoom() < LABEL_ZOOM_THRESHOLD);
    };
    cy.on("zoom", () => {
        if (!scheduled) {
            scheduled = true;
            requestAnimationFrame(update);
        }
    });
    update();
}

function wireGraphInteractions() {
    const cy = graphState.cy;

    cy.on("mouseover", "node", (evt) => {
        if (graphState.ego) return;
        const hood = evt.target.closedNeighborhood();
        cy.elements().difference(hood).addClass("faded");
        hood.nodes().removeClass("label-hidden");
        evt.target.connectedEdges().addClass("hl-edge");
    });
    cy.on("mouseout", "node", () => {
        if (graphState.ego) return;
        cy.elements().removeClass("faded hl-edge");
        if (cy.zoom() < LABEL_ZOOM_THRESHOLD) {
            cy.nodes().filter(n => n.data("freq") < 2 && n.data("centrality") < 0.05).addClass("label-hidden");
        }
    });

    cy.on("tap", "node", (evt) => {
        const hood = evt.target.closedNeighborhood();
        cy.elements().difference(hood).addClass("faded");
        hood.connectedEdges().addClass("hl-edge");
        graphState.ego = evt.target.id();
        document.getElementById("graph-ego-banner").classList.add("show");
        cy.animate({ fit: { eles: hood, padding: 70 }, duration: 300 });
    });

    cy.on("tap", (evt) => {
        if (evt.target === cy && graphState.ego) exitGraphEgo();
    });

    // double-click: open the summary of the concept's first source document
    cy.on("dbltap", "node", (evt) => {
        const docs = evt.target.data("docs") || [];
        if (docs.length) openSummaryFromKnowledgeMap(docs[0]);
    });
}

function exitGraphEgo() {
    if (!graphState.cy) return;
    graphState.ego = null;
    graphState.cy.elements().removeClass("faded hl-edge");
    document.getElementById("graph-ego-banner").classList.remove("show");
    graphState.cy.animate({ fit: { padding: 40 }, duration: 300 });
}

function applyGraphSearch(query) {
    const cy = graphState.cy;
    if (!cy) return;
    cy.nodes().removeClass("search-hit");
    if (!graphState.ego) cy.elements().removeClass("faded");
    if (!query) return;
    const hits = cy.nodes().filter(n => (n.data("label") || "").toLowerCase().includes(query));
    hits.addClass("search-hit");
    cy.elements().difference(hits.union(hits.connectedEdges()).union(hits.neighborhood())).addClass("faded");
}

function renderGraphLegend(payload) {
    const legendEl = document.getElementById("graph-legend");
    const commLabels = {};
    payload.communities.forEach(c => { commLabels[c.id] = c.label; });
    // count only nodes actually on screen so the legend reflects the view
    const counts = new Map();
    if (graphState.cy) {
        graphState.cy.nodes().forEach(n => {
            const c = n.data("community");
            if (c >= 0) counts.set(c, (counts.get(c) || 0) + 1);
        });
    }
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
    const items = top.map(([cid]) => {
        const color = GRAPH_PALETTE[cid % GRAPH_PALETTE.length];
        return `<span><span class="swatch" style="background:${color}"></span>${escapeHtml(commLabels[cid] || `cluster ${cid}`)}</span>`;
    });
    const rest = counts.size - top.length;
    if (rest > 0) items.push(`<span>+${rest} more</span>`);
    legendEl.innerHTML = items.join("");
}

function renderGraphDocFilter(docs) {
    const container = document.getElementById("graph-doc-filter");
    if (!docs.length) {
        container.innerHTML = '<p class="empty-state">No documents yet.</p>';
        return;
    }
    const selected = graphState.selectedDocs;
    container.innerHTML = docs.map(d => {
        const checked = selected === null || selected.has(d.id) ? "checked" : "";
        return `<label><input type="checkbox" data-doc-id="${d.id}" ${checked}>` +
               `<span class="doc-filter-title" title="${escapeHtml(d.title)}">${escapeHtml(d.title)}</span></label>`;
    }).join("");
    container.querySelectorAll("input[type='checkbox']").forEach(box => {
        box.addEventListener("change", () => {
            const all = [...container.querySelectorAll("input[type='checkbox']")];
            const checkedIds = all.filter(b => b.checked).map(b => b.dataset.docId);
            setGraphDocFilter(checkedIds.length === all.length ? null : new Set(checkedIds), { skipRerenderFilter: true });
        });
    });
}

function setGraphDocFilter(selectedDocs, opts = {}) {
    graphState.selectedDocs = selectedDocs;
    try {
        if (selectedDocs === null) localStorage.removeItem(graphDocFilterKey());
        else localStorage.setItem(graphDocFilterKey(), JSON.stringify([...selectedDocs]));
    } catch { /* storage unavailable */ }
    if (!opts.skipRerenderFilter && graphState.payload) {
        renderGraphDocFilter(graphState.payload.documents || []);
    }
    rebuildGraphView();
}

function renderCommunityPanel(payload) {
    const container = document.getElementById("community-list");
    const counts = {};
    payload.nodes.forEach(n => {
        if (n.community >= 0) counts[n.community] = (counts[n.community] || 0) + 1;
    });
    const rows = payload.communities.filter(c => counts[c.id]);
    if (!rows.length) {
        container.innerHTML = '<p class="empty-state">Refresh connections to detect clusters.</p>';
        return;
    }
    container.innerHTML = rows.map(c => {
        const color = GRAPH_PALETTE[c.id % GRAPH_PALETTE.length];
        return `<div class="cluster-row" data-community="${c.id}">` +
               `<span class="swatch" style="background:${color}"></span>${escapeHtml(c.label)}` +
               `<span class="count">${counts[c.id]}</span></div>`;
    }).join("");
    container.querySelectorAll(".cluster-row").forEach(row => {
        row.addEventListener("click", () => {
            const cy = graphState.cy;
            if (!cy) return;
            const cid = parseInt(row.dataset.community, 10);
            const isActive = row.classList.contains("active");
            container.querySelectorAll(".cluster-row").forEach(r => r.classList.remove("active"));
            cy.elements().removeClass("faded");
            if (!isActive) {
                row.classList.add("active");
                const members = cy.nodes().filter(n => n.data("community") === cid);
                cy.elements().difference(members.union(members.edgesWith(members))).addClass("faded");
            }
        });
    });
}

function renderGapPanel(payload) {
    const container = document.getElementById("gap-list");
    if (!payload.gaps.length) {
        container.innerHTML = '<p class="empty-state">No disconnected areas — your knowledge is well linked.</p>';
        return;
    }
    const commLabels = {};
    payload.communities.forEach(c => { commLabels[c.id] = c.label; });
    container.innerHTML = payload.gaps.map((g, i) => {
        const labelA = commLabels[g.a] || g.label_a;
        const labelB = commLabels[g.b] || g.label_b;
        return `<div class="gap-card">` +
               `<div class="gap-pair">${escapeHtml(labelA)} ↔ ${escapeHtml(labelB)}</div>` +
               `<div class="gap-why">${escapeHtml(g.suggestion || "")}</div>` +
               `<button class="btn btn-secondary" data-gap-index="${i}">질문하러 가기</button></div>`;
    }).join("");
    container.querySelectorAll("button[data-gap-index]").forEach(btn => {
        btn.addEventListener("click", () => {
            const g = payload.gaps[parseInt(btn.dataset.gapIndex, 10)];
            const labelA = commLabels[g.a] || g.label_a;
            const labelB = commLabels[g.b] || g.label_b;
            askBridgeQuestion(`${labelA}와(과) ${labelB}는 어떤 관련이 있나요?`);
        });
    });
}

function askBridgeQuestion(question) {
    window.location.hash = "chat";
    setTimeout(() => {
        const input = document.getElementById("chat-input");
        if (input) {
            input.value = question;
            input.focus();
        }
    }, 150);
}

// ── Loading ────────────────────────────────────────────────────────────────

function showLoading(text = "Processing...") {
    document.getElementById("loading-text").textContent = text;
    document.getElementById("loading-overlay").classList.remove("hidden");
}

function hideLoading() {
    document.getElementById("loading-overlay").classList.add("hidden");
}


function openSummaryFromKnowledgeMap(title) {
    window.location.hash = "summaries";

    setTimeout(() => {
        const cards = document.querySelectorAll(".summary-card");

        for (const card of cards) {
            const titleEl = card.querySelector(".summary-title");

            if (titleEl && titleEl.textContent.includes(title)) {
                card.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });

                card.classList.add("summary-highlight");

                setTimeout(() => {
                    card.classList.remove("summary-highlight");
                }, 2000);

                const body = card.querySelector(".summary-body");
                if (body) {
                    body.classList.add("open");
                }

                const toggle = card.querySelector(".summary-toggle");
                if (toggle) {
                    toggle.classList.add("open");
                }

                break;
            }
        }
    }, 150);
}

// ── Helpers ────────────────────────────────────────────────────────────────

function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ── Answer History ────────────────────────────────────────────────────────
async function loadAnswerHistory() {
    try {
        const res = await apiFetch(`/api/questions/history?session_id=default`);
        const data = await res.json();

        renderAnswerHistory(data.history || []);
    } catch (err) {
        console.error("Failed to load answer history:", err);
    }
}

function renderAnswerHistory(history) {
    const list = document.getElementById("history-list");
    if (!list) return;

    if (!history.length) {
        list.innerHTML = '<div class="empty-state">No history yet.</div>';
        return;
    }

    list.innerHTML = history.map(item => {
        if (item.type === "multiple_choice") {
            return `
                <div class="history-card">
                    <div class="history-type-badge">Multiple Choice</div>
                    <div class="history-score">
                        ${item.correct ? "Correct" : "Wrong"}
                    </div>
                    <div class="history-question">
                        ${escapeHtml(item.question_prompt || "")}
                    </div>
                    <div class="history-answer">
                        <strong>Your answer:</strong><br>
                        ${escapeHtml(item.selected_answer || "")}
                    </div>
                    <div class="history-feedback">
                        <strong>Correct answer:</strong><br>
                        ${escapeHtml(item.correct_answer || "")}
                    </div>
                    <div class="history-feedback">
                        <strong>Explanation:</strong><br>
                        ${escapeHtml(item.explanation || "")}
                    </div>
                    <div class="history-time">
                        ${escapeHtml(item.timestamp || "")}
                    </div>
                </div>
            `;
        }

        return `
            <div class="history-card">
                <div class="history-type-badge">Short Answer</div>
                <div class="history-score">Score: ${Math.round(item.score || 0)}%</div>
                <div class="history-question">
                    ${escapeHtml(item.question_prompt || "")}
                </div>
                <div class="history-answer">
                    <strong>Your answer:</strong><br>
                    ${escapeHtml(item.user_answer || "")}
                </div>
                <div class="history-feedback">
                    <strong>Feedback:</strong><br>
                    ${escapeHtml(item.feedback || "")}
                </div>
                <div class="history-time">
                    ${escapeHtml(item.timestamp || "")}
                </div>
            </div>
        `;
    }).join("");
}
