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
    topK: 4,              // max edges kept per concept (densest-first) to tame clutter
    layers: { triple: true, semantic: true, cooccur: false, trail: true },
    showBadges: false,    // source-document badges under concept labels (off = cleaner)
    dateCut: null,        // null = now (no temporal filter)
    dateRange: null,      // [minMs, maxMs] from payload
    selectedDocs: null,   // Set of doc ids whose concepts are visible; null = all
    highlightCommunity: null, // community id highlighted from the Clusters panel
    ego: null,
};

// register the fcose layout extension once (clustered force-directed layout)
if (window.cytoscape && window.cytoscapeFcose) {
    try { window.cytoscape.use(window.cytoscapeFcose); } catch (e) { /* already registered */ }
}

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

    const SVG_MAX = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>';
    const SVG_MIN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>';
    const maxBtn = document.getElementById("graph-maximize");
    const reflow = () => setTimeout(() => {
        if (graphState.cy) { graphState.cy.resize(); graphState.cy.fit(undefined, 45); }
    }, 90);
    const setMaximized = (on) => {
        const layout = document.querySelector(".connections-layout");
        layout.classList.toggle("maximized", on);
        if (maxBtn) {
            maxBtn.innerHTML = on ? SVG_MIN : SVG_MAX;
            maxBtn.title = on ? "원래 크기로 (ESC)" : "전체 화면으로 크게 보기 (ESC로 복귀)";
        }
        reflow();
    };
    if (maxBtn) {
        maxBtn.addEventListener("click", () =>
            setMaximized(!document.querySelector(".connections-layout").classList.contains("maximized")));
    }
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && document.querySelector(".connections-layout")?.classList.contains("maximized")) {
            setMaximized(false);
        }
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

    const handle = document.querySelector(".panel-handle");
    const layout = document.querySelector(".connections-layout");
    if (handle && layout) {
        handle.addEventListener("click", () => {
            const collapsed = layout.classList.toggle("panel-collapsed");
            handle.textContent = collapsed ? "‹" : "›";
            setTimeout(() => { if (graphState.cy) graphState.cy.resize(); }, 260);
        });
    }
}

async function renderConnections() {
    const emptyEl = document.getElementById("connections-empty");
    const wrapEl = document.getElementById("cy-wrap");

    let payload = { nodes: [], edges: [], communities: [], gaps: [], documents: [], learning_questions: [] };
    try {
        const res = await apiFetch("/api/graph");
        if (res.ok) payload = await res.json();
    } catch (err) {
        console.error("Failed to load graph:", err);
    }
    graphState.payload = payload;

    const stamps = payload.nodes.map(n => Date.parse(n.created_at)).filter(t => !isNaN(t));
    graphState.dateRange = stamps.length ? [Math.min(...stamps), Math.max(...stamps)] : null;

    try {
        const saved = localStorage.getItem(graphDocFilterKey());
        graphState.selectedDocs = saved ? new Set(JSON.parse(saved)) : null;
    } catch { graphState.selectedDocs = null; }
    graphState.highlightCommunity = null;

    const hasConcepts = payload.nodes.length > 0;
    wrapEl.style.display = hasConcepts ? "block" : "none";
    emptyEl.style.display = hasConcepts ? "none" : "block";

    renderGraphDocFilter(payload.documents || []);
    renderCommunityPanel(payload);
    renderLearningPanel(payload);
    if (hasConcepts) rebuildGraphView();
}

// Short badge for a document title: leading chapter number or first word.
function docBadge(title) {
    const m = /^(\d+)\./.exec(title || "");
    if (m) return m[1];
    return (title || "?").split(/\s+/)[0].slice(0, 6);
}

// Full concept graph: every concept is a node; the fcose layout pulls each
// community into its own spatial region. Edges are pruned to top-K per concept
// so dense within-cluster cliques don't form a hairball.
function graphCyElements(payload) {
    const { topK, layers, dateCut, selectedDocs, showBadges } = graphState;
    const inDate = (iso) => !dateCut || !iso || isNaN(Date.parse(iso)) || iso <= dateCut;
    const passes = (n) => inDate(n.created_at) &&
        (selectedDocs === null || (n.doc_ids || []).some(id => selectedDocs.has(id)));

    const visible = payload.nodes.filter(passes);
    const ids = new Set(visible.map(n => n.id));
    const nodes = visible.map(n => {
        const badge = (n.docs || []).map(docBadge).join("·");
        return { data: { ...n, type: "concept", lblH: "center", lblV: "bottom",
            display: showBadges && badge ? `${n.label}\n[${badge}]` : n.label } };
    });

    // candidate edges among visible nodes, per active layers
    const candidates = payload.edges.filter(e =>
        ids.has(e.source) && ids.has(e.target) && layers[e.kind] !== false);

    // keep each node's strongest topK edges (typed/triple always kept)
    const sorted = candidates.slice().sort((a, b) => (b.weight || 0) - (a.weight || 0));
    const perNode = new Map();
    const kept = [];
    for (const e of sorted) {
        if (e.kind === "triple") { kept.push(e); continue; }
        const cs = perNode.get(e.source) || 0;
        const ct = perNode.get(e.target) || 0;
        if (cs < topK || ct < topK) {
            kept.push(e);
            perNode.set(e.source, cs + 1);
            perNode.set(e.target, ct + 1);
        }
    }
    const edges = kept.map((e, i) => ({ data: {
        id: `e${i}`, source: e.source, target: e.target, kind: e.kind,
        weight: e.weight || 1, category: e.category || "",
        label: e.label || "", shared_docs: e.shared_docs || [],
        reason: edgeReason(e),
    }}));

    // drop isolated nodes (no surviving edge)
    const connected = new Set();
    edges.forEach(e => { connected.add(e.data.source); connected.add(e.data.target); });
    let keptNodes = nodes.filter(n => connected.has(n.data.id));

    // keep only the main connected component — small satellites floating off to
    // the side (single concepts, 2-3 node fragments) just add visual noise
    const adj = new Map();
    keptNodes.forEach(n => adj.set(n.data.id, []));
    edges.forEach(e => {
        adj.get(e.data.source)?.push(e.data.target);
        adj.get(e.data.target)?.push(e.data.source);
    });
    const seen = new Set();
    let biggest = [];
    for (const n of keptNodes) {
        if (seen.has(n.data.id)) continue;
        const stack = [n.data.id]; const members = [];
        while (stack.length) {
            const x = stack.pop();
            if (seen.has(x)) continue;
            seen.add(x); members.push(x);
            (adj.get(x) || []).forEach(y => { if (!seen.has(y)) stack.push(y); });
        }
        if (members.length > biggest.length) biggest = members;
    }
    const keepId = new Set(biggest);
    keptNodes = keptNodes.filter(n => keepId.has(n.data.id));
    const finalEdges = edges.filter(e => keepId.has(e.data.source) && keepId.has(e.data.target));
    return [...keptNodes, ...finalEdges];
}

// Model-space length of an edge (zoom-independent), for length-aware label sizing.
function edgeLen(ele) {
    const s = ele.source().position(), t = ele.target().position();
    return Math.hypot(t.x - s.x, t.y - s.y);
}

// Human-readable reason an edge exists — the "why" behind every connection.
function edgeReason(e) {
    switch (e.kind) {
        case "triple": return e.label ? `relation: ${e.label}` : "typed relation";
        case "semantic": return e.label ? `similar: ${e.label}` : "semantically similar";
        case "trail": return "discussed together in a Q&A";
        case "cooccur": return e.shared_docs && e.shared_docs.length
            ? `appear together in ${e.shared_docs.join(", ")}` : "appear in the same document";
        default: return e.kind;
    }
}

function graphLayout() {
    if (window.cytoscapeFcose) {
        return {
            name: "fcose", quality: "proof", animate: false, randomize: true,
            // same-community edges pull a little tighter so clusters read as loose
            // groupings — by position only, no boxes (Obsidian-like calm layout)
            idealEdgeLength: (edge) =>
                edge.source().data("community") === edge.target().data("community") ? 55 : 130,
            edgeElasticity: 0.4,
            nodeRepulsion: 5500, gravity: 0.12, gravityRange: 4.0,
            numIter: 2500, packComponents: true, nodeSeparation: 95, padding: 50,
        };
    }
    return { name: "cose", animate: false, nodeRepulsion: 200000, idealEdgeLength: 90,
             gravity: 0.2, padding: 40, componentSpacing: 140, nodeOverlap: 30 };
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
            // minimal: small uniform dots, soft community color, thin links
            { selector: "node[type='concept']", style: {
                "background-color": (ele) => GRAPH_PALETTE[Math.max(ele.data("community"), 0) % GRAPH_PALETTE.length],
                "background-opacity": 0.92,
                "width": (ele) => 7 + Math.min(ele.data("freq"), 4) * 2.5 + ele.data("centrality") * 14,
                "height": (ele) => 7 + Math.min(ele.data("freq"), 4) * 2.5 + ele.data("centrality") * 14,
                "label": "data(display)", "color": "#9aa0ad", "font-size": 8.5,
                "text-halign": "data(lblH)", "text-valign": "data(lblV)",
                "text-margin-x": (ele) => { const h = ele.data("lblH"); return h === "right" ? 7 : h === "left" ? -7 : 0; },
                "text-margin-y": (ele) => { const v = ele.data("lblV"); return v === "bottom" ? 4 : v === "top" ? -4 : 0; },
                "text-max-width": "110px", "text-wrap": "wrap",
                "border-width": 0,
                "transition-property": "opacity, background-opacity", "transition-duration": "0.12s",
            }},
            { selector: "node[kind='entity']", style: { "shape": "round-rectangle" }},
            { selector: "edge", style: { "curve-style": "haystack", "haystack-radius": 0,
                "line-color": "#2f3340", "width": 0.8, "opacity": 0.6 }},
            { selector: "edge[kind='semantic']", style: { "line-color": "#5a6178", "width": 1 }},
            { selector: "edge[kind='triple']", style: {
                "line-color": "#5d6470", "width": 1,
                "target-arrow-shape": "triangle", "arrow-scale": 0.55, "curve-style": "bezier",
                "target-arrow-color": "#5d6470" }},
            { selector: "edge[kind='trail']", style: { "line-style": "dotted", "line-color": "#3f4a63", "width": 1, "curve-style": "bezier" }},
            { selector: "edge[kind='cooccur']", style: { "line-color": "#262a35", "width": 0.7 }},
            { selector: ".label-hidden", style: { "text-opacity": 0 }},
            { selector: ".dimmed", style: { "opacity": 0.12, "text-opacity": 0.05 }},
            { selector: ".faded", style: { "opacity": 0.06, "text-opacity": 0.02 }},
            { selector: ".hot", style: { "background-opacity": 1, "color": "#e7eaf0", "z-index": 20 }},
            // highlighted edges: emphasise the line only — the "why" lives in the
            // inspector, so no on-canvas labels to overlap (clutter killer)
            { selector: ".hl-edge", style: {
                "line-color": "#c8a23f", "width": 1.6, "opacity": 0.95, "z-index": 19 }},
            // a single hovered edge may show its reason; font + width scale with the
            // edge length so a short line never gets a label that overruns its nodes
            { selector: "edge.edge-reason", style: {
                "label": "data(reason)",
                "font-size": (ele) => { const L = edgeLen(ele); return Math.max(6, Math.min(10, L / 26)); },
                "text-max-width": (ele) => `${Math.max(40, edgeLen(ele) * 0.66)}px`,
                "color": "#e8b34b", "text-background-color": "#0c0e13",
                "text-background-opacity": 0.92, "text-background-padding": "3px",
                "text-rotation": "autorotate", "text-wrap": "wrap",
                "line-color": "#c8a23f", "width": 2, "opacity": 1, "z-index": 30, "curve-style": "bezier" }},
            { selector: "node.search-hit", style: { "border-width": 2, "border-color": "#e8b34b", "background-opacity": 1 }},
            { selector: "node.selected-node", style: { "border-width": 2, "border-color": "#e8b34b", "background-opacity": 1, "color": "#e7eaf0" }},
        ],
        layout: graphLayout(),
        wheelSensitivity: 0.2,
    });
    wireGraphInteractions();
    wireZoomAdaptiveLabels();
    renderGraphLegend(graphState.payload);
    // base positions are captured lazily on the first ego click (see tap handler),
    // so a fresh rebuild forgets any stale snapshot
    graphState.basePositions = null;
    if (graphState.highlightCommunity !== null) highlightCommunity(graphState.highlightCommunity);
    applyGraphSearch(document.getElementById("graph-search").value.trim().toLowerCase());
}

// Point each neighbour's label away from the focused node (outward), so labels
// fan out around the ring instead of piling up near the centre.
function layoutLabelsOutward(center, hood) {
    const c = center.position();
    hood.nodes().forEach(nb => {
        if (nb.same(center)) { nb.data("lblH", "center"); nb.data("lblV", "bottom"); return; }
        const p = nb.position(); const dx = p.x - c.x, dy = p.y - c.y;
        if (Math.abs(dx) > Math.abs(dy)) {
            nb.data("lblH", dx >= 0 ? "right" : "left"); nb.data("lblV", "center");
        } else {
            nb.data("lblH", "center"); nb.data("lblV", dy >= 0 ? "bottom" : "top");
        }
    });
}

// Below this zoom only hub concepts keep labels (overview readability).
const LABEL_ZOOM_THRESHOLD = 0.85;

function wireZoomAdaptiveLabels() {
    const cy = graphState.cy;
    const minor = cy.nodes().filter(n => n.data("freq") < 2 && n.data("centrality") < 0.06);
    let scheduled = false;
    const update = () => {
        scheduled = false;
        minor.toggleClass("label-hidden", cy.zoom() < LABEL_ZOOM_THRESHOLD);
    };
    cy.on("zoom", () => { if (!scheduled) { scheduled = true; requestAnimationFrame(update); } });
    update();
}

function wireGraphInteractions() {
    const cy = graphState.cy;

    // hover: gently surface the node's neighbourhood + label the connections
    cy.on("mouseover", "node[type='concept']", (evt) => {
        if (graphState.ego) return;
        const hood = evt.target.closedNeighborhood();
        cy.elements().difference(hood).addClass("dimmed");
        hood.nodes().removeClass("label-hidden").addClass("hot");
        evt.target.connectedEdges().addClass("hl-edge");
    });
    cy.on("mouseout", "node[type='concept']", () => {
        if (graphState.ego) return;
        cy.elements().removeClass("dimmed hl-edge hot");
        applyZoomLabels();
        if (graphState.highlightCommunity !== null) highlightCommunity(graphState.highlightCommunity);
    });

    // click: focus the node, spread its neighbours on a ring, explain in inspector
    cy.on("tap", "node[type='concept']", (evt) => {
        const node = evt.target;
        // snapshot the current (force-directed) positions BEFORE the ring layout
        // disturbs them, so closing the focus can restore the map exactly
        if (!graphState.basePositions) {
            graphState.basePositions = {};
            cy.nodes().forEach(n => { graphState.basePositions[n.id()] = { ...n.position() }; });
        }
        const hood = node.closedNeighborhood();
        cy.elements().removeClass("dimmed faded hl-edge hot selected-node");
        cy.elements().difference(hood).addClass("faded");
        node.connectedEdges().addClass("hl-edge");
        hood.nodes().removeClass("label-hidden").addClass("hot");
        node.addClass("selected-node");
        graphState.ego = node.id();
        showConceptInspector(node);

        // concentric re-layout of just the neighbourhood: even angular spacing
        // means the dots (and so their labels) no longer overlap near the hub
        const others = hood.nodes().not(node);
        const spacing = Math.max(46, Math.min(120, 1100 / Math.max(others.length, 1)));
        hood.layout({
            name: "concentric",
            concentric: (n) => (n.same(node) ? 10 : 1),
            levelWidth: () => 1,
            minNodeSpacing: spacing,
            animate: true, animationDuration: 350, fit: true, padding: 90,
            startAngle: -Math.PI / 2,
        }).run();
        graphState.cy.one("layoutstop", () => layoutLabelsOutward(node, hood));
    });

    cy.on("tap", (evt) => { if (evt.target === cy && graphState.ego) exitGraphEgo(); });

    // hovering a single edge reveals just its reason — but a dimmed/faded edge
    // (outside the current focus) stays quiet so only the relevant line responds
    cy.on("mouseover", "edge", (evt) => {
        if (evt.target.hasClass("faded") || evt.target.hasClass("dimmed")) return;
        evt.target.addClass("edge-reason");
    });
    cy.on("mouseout", "edge", (evt) => evt.target.removeClass("edge-reason"));

    cy.on("dbltap", "node[type='concept']", (evt) => {
        const docs = evt.target.data("docs") || [];
        if (docs.length) openSummaryFromKnowledgeMap(docs[0]);
    });
}

function applyZoomLabels() {
    const cy = graphState.cy;
    if (!cy) return;
    const small = cy.zoom() < LABEL_ZOOM_THRESHOLD;
    cy.nodes().filter(n => n.data("freq") < 2 && n.data("centrality") < 0.06)
        .toggleClass("label-hidden", small);
}

// Inspector: explains WHAT a concept is and WHY it links to its neighbours.
function showConceptInspector(node) {
    const box = document.getElementById("graph-inspector");
    if (!box) return;
    const d = node.data();
    const payload = graphState.payload || {};
    const commLabel = (payload.communities || []).find(c => c.id === d.community);
    const docs = (d.docs || []);
    const cy = graphState.cy;

    const conns = node.connectedEdges().map(e => {
        const other = e.source().id() === node.id() ? e.target() : e.source();
        return { label: other.data("label"), reason: e.data("reason"), kind: e.data("kind") };
    }).sort((a, b) => a.kind.localeCompare(b.kind));

    const connHtml = conns.length
        ? conns.map(c => `<li><span class="ins-other">${escapeHtml(c.label)}</span>` +
            `<span class="ins-reason">${escapeHtml(c.reason)}</span></li>`).join("")
        : '<li class="empty-state">No connections.</li>';

    box.innerHTML =
        `<div class="ins-head">` +
        `<span class="ins-dot" style="background:${GRAPH_PALETTE[Math.max(d.community,0)%GRAPH_PALETTE.length]}"></span>` +
        `<span class="ins-title">${escapeHtml(d.label)}</span>` +
        `<button class="ins-close" type="button" aria-label="Close">×</button></div>` +
        `<div class="ins-meta">${d.kind === "entity" ? (d.entity_type || "entity") : "concept"}` +
        (commLabel ? ` · ${escapeHtml(commLabel.label)}` : "") +
        (docs.length ? ` · in ${docs.length} doc${docs.length > 1 ? "s" : ""}` : "") + `</div>` +
        (docs.length ? `<div class="ins-docs">From: ${docs.map(escapeHtml).join(", ")}</div>` : "") +
        `<div class="ins-conn-title">Connected to (why):</div>` +
        `<ul class="ins-conn">${connHtml}</ul>` +
        (docs.length ? `<button class="btn btn-secondary ins-open" type="button">Open document</button>` : "");

    const openBtn = box.querySelector(".ins-open");
    if (openBtn) openBtn.addEventListener("click", () => openSummaryFromKnowledgeMap(docs[0]));
    const closeBtn = box.querySelector(".ins-close");
    if (closeBtn) closeBtn.addEventListener("click", exitGraphEgo);
    box.classList.add("show");
}

function exitGraphEgo() {
    if (!graphState.cy) return;
    const cy = graphState.cy;
    graphState.ego = null;
    cy.elements().removeClass("faded dimmed hl-edge hot selected-node");
    cy.nodes().forEach(n => { n.data("lblH", "center"); n.data("lblV", "bottom"); });
    document.getElementById("graph-ego-banner").classList.remove("show");
    const box = document.getElementById("graph-inspector");
    if (box) box.classList.remove("show");
    // restore the original force-directed positions disturbed by the ring layout
    const base = graphState.basePositions;
    if (base) {
        cy.nodes().forEach(n => { const p = base[n.id()]; if (p) n.animate({ position: p }, { duration: 320 }); });
        setTimeout(() => cy.animate({ fit: { padding: 45 }, duration: 300 }), 60);
    } else {
        cy.animate({ fit: { padding: 45 }, duration: 300 });
    }
    applyZoomLabels();
    if (graphState.highlightCommunity !== null) highlightCommunity(graphState.highlightCommunity);
}

function highlightCommunity(cid) {
    const cy = graphState.cy;
    if (!cy) return;
    cy.elements().removeClass("faded");
    if (cid === null) return;
    const members = cy.nodes().filter(n => n.data("community") === cid);
    cy.elements().difference(members.union(members.edgesWith(members))).addClass("faded");
}

function applyGraphSearch(query) {
    const cy = graphState.cy;
    if (!cy) return;
    cy.nodes().removeClass("search-hit");
    if (!graphState.ego && graphState.highlightCommunity === null) cy.elements().removeClass("faded");
    if (!query) return;
    const hits = cy.nodes().filter(n => (n.data("label") || "").toLowerCase().includes(query));
    hits.addClass("search-hit");
    cy.elements().difference(hits.union(hits.connectedEdges()).union(hits.neighborhood())).addClass("faded");
}

function renderGraphLegend(payload) {
    const legendEl = document.getElementById("graph-legend");
    const commLabels = {};
    payload.communities.forEach(c => { commLabels[c.id] = c.label; });
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
        const active = graphState.highlightCommunity === c.id ? "active" : "";
        return `<div class="cluster-row ${active}" data-community="${c.id}">` +
               `<span class="swatch" style="background:${color}"></span>${escapeHtml(c.label)}` +
               `<span class="count">${counts[c.id]}</span></div>`;
    }).join("");
    container.querySelectorAll(".cluster-row").forEach(row => {
        row.addEventListener("click", () => {
            const cid = parseInt(row.dataset.community, 10);
            graphState.highlightCommunity = graphState.highlightCommunity === cid ? null : cid;
            renderCommunityPanel(payload);
            const cy = graphState.cy;
            if (!cy) return;
            if (graphState.highlightCommunity === null) {
                cy.elements().removeClass("faded");
            } else {
                highlightCommunity(cid);
                const members = cy.nodes().filter(n => n.data("community") === cid);
                if (members.length) cy.animate({ fit: { eles: members, padding: 80 }, duration: 350 });
            }
        });
    });
}

// Study questions grounded in cluster topics; click to ask in Chat.
function renderLearningPanel(payload) {
    const container = document.getElementById("learning-list");
    const questions = payload.learning_questions || [];
    if (!questions.length) {
        container.innerHTML = '<p class="empty-state">Refresh connections to get study questions.</p>';
        return;
    }
    container.innerHTML = questions.map((q, i) =>
        `<div class="learn-card" data-i="${i}">` +
        (q.topic ? `<div class="learn-topic">${escapeHtml(q.topic)}</div>` : "") +
        `<div class="learn-q">${escapeHtml(q.question)}</div></div>`
    ).join("");
    container.querySelectorAll(".learn-card").forEach(card => {
        card.addEventListener("click", () => {
            askInChat(questions[parseInt(card.dataset.i, 10)].question);
        });
    });
}

function askInChat(question) {
    window.location.hash = "chat";
    setTimeout(() => {
        const input = document.getElementById("chat-input");
        if (input) { input.value = question; input.focus(); }
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
