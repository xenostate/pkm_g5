/* ── PKM Dashboard App ──────────────────────────────────────────────────── */

const API = "";  // same origin

// ── State ──────────────────────────────────────────────────────────────────

let documents = [];
let sessionId = sessionStorage.getItem("pkm_session") || crypto.randomUUID();
sessionStorage.setItem("pkm_session", sessionId);
let questionDocuments = [];
let activeQuestionDocId = null;
let activeQuestion = null;
let lastQuestionResult = null;
const knowledgeMapState = {
    scale: 1,
    minScale: 0.75,
    maxScale: 2.5,
    step: 0.2,
    payload: null,
    panX: 0,
    panY: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    hoverTargets: [],
    showSimilarity: true,
    showConcepts: true,
    showTopics: true,
};
const KNOWLEDGE_MAP_STORAGE_KEY = "pkm_knowledge_map_view";
let knowledgeMapAutoRefreshTried = false;

restoreKnowledgeMapView();

// ── Init ───────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    initRouter();
    initUpload();
    initSearch();
    initChat();
    initQuestions();
    initConnections();
    loadDocuments();
    loadStats();
});

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
            const res = await fetch(`${API}/api/documents/upload-pdf`, { method: "POST", body: form });
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
        const res = await fetch(`${API}/api/documents/add-url`, {
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
        const res = await fetch(`${API}/api/documents/add-text`, {
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
        const res = await fetch(`${API}/api/documents`);
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
        const res = await fetch(`${API}/api/documents/${docId}`, { method: "DELETE" });
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
        const res = await fetch(`${API}/api/stats`);
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
        const res = await fetch(`${API}/api/search`, {
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

    btn.addEventListener("click", () => sendChat());
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChat();
    });
}

async function sendChat() {
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) return;

    input.value = "";
    appendChatMsg("user", message);

    // Show typing indicator
    const typingEl = appendChatMsg("assistant", "Thinking...");
    typingEl.style.opacity = "0.5";

    try {
        const res = await fetch(`${API}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId }),
        });
        const data = await res.json();

        typingEl.remove();
        appendChatAnswer(data);
        loadStats();
    } catch (err) {
        typingEl.remove();
        appendChatMsg("assistant", `Error: ${err.message}`);
    }
}

function appendChatMsg(role, text) {
    const container = document.getElementById("chat-messages");
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

function appendChatAnswer(data) {
    const container = document.getElementById("chat-messages");

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
        const res = await fetch(`${API}/api/questions`);
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
            <div class="question-feedback-status">Submitted</div>
            <div class="question-explanation">
                <strong>Sample answer:</strong><br>
                ${escapeHtml(question.sample_answer || "No sample answer available.")}
            </div>
            <div class="question-explanation">
                ${escapeHtml(question.explanation || "")}
            </div>
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

    lastQuestionResult = {
        submitted: true,
    };

    renderQuestionsPanel();
}

async function generateQuestionsForActiveDoc() {
    if (!activeQuestionDocId) return;

    const questionType = document.getElementById("question-type-select").value;

    showLoading("Generating study questions...");
    try {
        const res = await fetch(`${API}/api/questions/generate`, {
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
        const res = await fetch(`${API}/api/questions/answer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: sessionId,
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
        const res = await fetch(`${API}/api/questions/next`, {
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
    const canvas = document.getElementById("connections-canvas");
    const tooltip = document.getElementById("connections-tooltip");

    document.getElementById("refresh-connections-btn").addEventListener("click", async () => {
        showLoading("Computing knowledge connections...");
        try {
            const res = await fetch(`${API}/api/connections/refresh`, { method: "POST" });
            if (!res.ok) throw new Error("Refresh failed");
            const data = await res.json();
            await loadDocuments();
            renderConnections();
            if (data.concepts_backfilled) {
                alert(`Refreshed connections and backfilled concepts for ${data.concepts_backfilled} document(s).`);
            }
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
        hideLoading();
    });

    document.getElementById("connections-zoom-in").addEventListener("click", () => {
        setKnowledgeMapScale(knowledgeMapState.scale + knowledgeMapState.step);
    });

    document.getElementById("connections-zoom-out").addEventListener("click", () => {
        setKnowledgeMapScale(knowledgeMapState.scale - knowledgeMapState.step);
    });

    document.getElementById("connections-zoom-reset").addEventListener("click", () => {
        setKnowledgeMapScale(1);
    });

    document.getElementById("toggle-similarity").addEventListener("click", () => {
        toggleKnowledgeMapLayer("showSimilarity", "toggle-similarity");
    });

    document.getElementById("toggle-concepts").addEventListener("click", () => {
        toggleKnowledgeMapLayer("showConcepts", "toggle-concepts");
    });

    document.getElementById("toggle-topics").addEventListener("click", () => {
        toggleKnowledgeMapLayer("showTopics", "toggle-topics");
    });

    canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        const delta = event.deltaY < 0 ? knowledgeMapState.step : -knowledgeMapState.step;
        setKnowledgeMapScale(knowledgeMapState.scale + delta);
    }, { passive: false });

    canvas.addEventListener("mousedown", (event) => {
        knowledgeMapState.isDragging = true;
        knowledgeMapState.dragStartX = event.clientX;
        knowledgeMapState.dragStartY = event.clientY;
        canvas.classList.add("dragging");
        hideConnectionsTooltip();
    });

    canvas.addEventListener("mousemove", (event) => {
        if (knowledgeMapState.isDragging) {
            const dx = event.clientX - knowledgeMapState.dragStartX;
            const dy = event.clientY - knowledgeMapState.dragStartY;
            knowledgeMapState.dragStartX = event.clientX;
            knowledgeMapState.dragStartY = event.clientY;
            knowledgeMapState.panX += dx;
            knowledgeMapState.panY += dy;
            persistKnowledgeMapView();
            drawConnectionsMap();
            return;
        }

        const hit = findKnowledgeMapHoverTarget(event);
        if (!hit) {
            canvas.style.cursor = knowledgeMapState.scale > 1 ? "grab" : "default";
            hideConnectionsTooltip();
            return;
        }

        canvas.style.cursor = "pointer";
        showConnectionsTooltip(hit, event.clientX, event.clientY, tooltip);
    });

    const stopDragging = () => {
        knowledgeMapState.isDragging = false;
        canvas.classList.remove("dragging");
        canvas.style.cursor = knowledgeMapState.scale > 1 ? "grab" : "default";
    };

    canvas.addEventListener("mouseup", stopDragging);
    canvas.addEventListener("mouseleave", () => {
        stopDragging();
        hideConnectionsTooltip();
    });
    window.addEventListener("mouseup", stopDragging);

    updateKnowledgeMapZoomLabel();
    syncKnowledgeMapToggleButtons();
}

async function renderConnections() {
    const canvas = document.getElementById("connections-canvas");
    const emptyEl = document.getElementById("connections-empty");
    const legendEl = document.getElementById("connections-legend");
    const topicListEl = document.getElementById("topic-correlation-list");
    const ctx = canvas.getContext("2d");

    let connections = [];
    let conceptLinks = [];
    let knowledgeBase = { documents: {}, links: [] };
    try {
        const [connectionsRes, kbRes] = await Promise.all([
            fetch(`${API}/api/connections`),
            fetch(`${API}/api/knowledge-base`),
        ]);
        const connectionsData = await connectionsRes.json();
        const kbData = await kbRes.json();
        connections = connectionsData.connections || [];
        conceptLinks = kbData.links || [];
        knowledgeBase = kbData || knowledgeBase;
    } catch (err) {
        console.error("Failed to load connections:", err);
    }

    const orbitTopicNodes = buildOrbitTopicNodes(knowledgeBase);
    const topicBridges = buildTopicBridges(orbitTopicNodes);
    const effectiveConceptLinks = conceptLinks.length ? conceptLinks : buildSharedConceptLinksFromTopicBridges(topicBridges);
    renderTopicCorrelations(topicBridges, topicListEl);

    const needsAutoRefresh = documents.length > 0 && !knowledgeMapAutoRefreshTried && (
        connections.length === 0 ||
        orbitTopicNodes.length === 0 ||
        (knowledgeMapState.showSimilarity && documents.length > 1 && connections.length < 1)
    );

    if (needsAutoRefresh) {
        knowledgeMapAutoRefreshTried = true;
        try {
            await fetch(`${API}/api/connections/refresh`, { method: "POST" });
            await loadDocuments();
            return renderConnections();
        } catch (err) {
            console.error("Automatic knowledge-map refresh failed:", err);
        }
    }

    knowledgeMapState.payload = {
        connections,
        conceptLinks: effectiveConceptLinks,
        orbitTopicNodes,
        topicBridges,
        knowledgeBase,
    };

    if (!documents.length) {
        canvas.style.display = "none";
        legendEl.style.display = "none";
        emptyEl.style.display = "block";
        return;
    }

    canvas.style.display = "block";
    legendEl.style.display = "flex";
    emptyEl.style.display = "none";

    // Resize canvas
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = Math.max(rect.height, 400);
    drawConnectionsMap();
}

function drawConnectionsMap() {
    const canvas = document.getElementById("connections-canvas");
    const ctx = canvas.getContext("2d");
    const payload = knowledgeMapState.payload;
    if (!payload) return;

    const { connections, conceptLinks, orbitTopicNodes, topicBridges, knowledgeBase } = payload;

    // Build node positions (simple circle layout)
    const nodes = {};
    documents.forEach((doc, i) => {
        const angle = (2 * Math.PI * i) / documents.length - Math.PI / 2;
        const rx = canvas.width * (documents.length === 1 ? 0 : 0.34);
        const ry = canvas.height * (documents.length === 1 ? 0 : 0.32);
        nodes[doc.doc_id] = {
            x: canvas.width / 2 + rx * Math.cos(angle),
            y: canvas.height / 2 + ry * Math.sin(angle),
            title: doc.title,
            type: doc.source_type,
            radius: 30,
        };
    });

    orbitTopicNodes.forEach(topicNode => {
        const docNode = nodes[topicNode.docId];
        if (!docNode) return;
        const orbitRadius = docNode.radius + 30 + topicNode.rank * 3;
        const angle = topicNode.angle;
        topicNode.x = docNode.x + orbitRadius * Math.cos(angle);
        topicNode.y = docNode.y + orbitRadius * Math.sin(angle);
    });

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.translate(knowledgeMapState.panX, knowledgeMapState.panY);
    ctx.scale(knowledgeMapState.scale, knowledgeMapState.scale);
    ctx.translate(-canvas.width / 2, -canvas.height / 2);

    if (knowledgeMapState.showSimilarity) {
        const strongestConnections = connections
            .slice()
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, Math.max(4, documents.length + 1));

        const seen = new Set();
        strongestConnections.forEach(conn => {
            const key = [conn.from_doc_id, conn.to_doc_id].sort().join("-");
            if (seen.has(key)) return;
            seen.add(key);

            const from = nodes[conn.from_doc_id];
            const to = nodes[conn.to_doc_id];
            if (!from || !to) return;

            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.strokeStyle = `rgba(255, 255, 255, ${Math.min(Math.max(conn.similarity, 0.18), 0.5)})`;
            ctx.lineWidth = Math.max(1, conn.similarity * 4);
            ctx.stroke();

            const mx = (from.x + to.x) / 2;
            const my = (from.y + to.y) / 2;
            ctx.fillStyle = "rgba(156, 163, 175, 0.8)";
            ctx.font = "10px Nunito, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText((conn.similarity * 100).toFixed(0) + "%", mx, my - 4);
        });
    }

    if (knowledgeMapState.showConcepts) {
        conceptLinks.forEach((link, index) => {
            const from = nodes[link.from];
            const to = nodes[link.to];
            if (!from || !to) return;

            ctx.save();
            ctx.beginPath();
            ctx.setLineDash([6, 5]);
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.strokeStyle = "rgba(255, 196, 61, 0.85)";
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();

            const mx = (from.x + to.x) / 2;
            const my = (from.y + to.y) / 2;
            const concepts = Array.isArray(link.concept) ? link.concept.join(", ") : "";
            const label = concepts.length > 28 ? concepts.substring(0, 26) + "..." : concepts;
            if (!label) return;

            const offset = index % 2 === 0 ? 12 : 24;
            ctx.fillStyle = "rgba(255, 196, 61, 0.95)";
            ctx.font = "11px Nunito, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(label, mx, my + offset);
        });
    }

    if (knowledgeMapState.showTopics) {
        orbitTopicNodes.forEach(topicNode => {
            const docNode = nodes[topicNode.docId];
            if (!docNode) return;
            ctx.save();
            ctx.beginPath();
            ctx.setLineDash([2, 6]);
            ctx.moveTo(docNode.x, docNode.y);
            ctx.lineTo(topicNode.x, topicNode.y);
            ctx.strokeStyle = "rgba(251, 191, 36, 0.28)";
            ctx.lineWidth = 1.2;
            ctx.stroke();
            ctx.restore();
        });

        topicBridges.forEach(bridge => {
            const from = bridge.from;
            const to = bridge.to;
            if (!from || !to) return;

            ctx.save();
            ctx.beginPath();
            ctx.setLineDash([6, 5]);
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.strokeStyle = `rgba(251, 191, 36, ${Math.min(0.88, 0.32 + bridge.score * 0.45)})`;
            ctx.lineWidth = 1 + bridge.score * 2.4;
            ctx.stroke();
            ctx.restore();

            const mx = (from.x + to.x) / 2;
            const my = (from.y + to.y) / 2;
            ctx.fillStyle = "rgba(255, 221, 120, 0.92)";
            ctx.font = "10px Nunito, sans-serif";
            ctx.textAlign = "center";
            const label = bridge.label.length > 26 ? bridge.label.substring(0, 24) + "..." : bridge.label;
            ctx.fillText(label, mx, my - 6);
        });
    }

    // Draw nodes
    const typeColors = { pdf: "#f87171", url: "#60a5fa", text: "#4ade80" };

    Object.values(nodes).forEach(node => {
        // Outer glow
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius + 8, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.04)";
        ctx.fill();

        // Main circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = typeColors[node.type] || "#6366f1";
        ctx.fill();
        ctx.strokeStyle = "#1c1f2e";
        ctx.lineWidth = 3;
        ctx.stroke();

        // Label
        ctx.fillStyle = "#e4e4e7";
        ctx.font = "13px Nunito, sans-serif";
        ctx.textAlign = "center";
        const label = node.title.length > 22 ? node.title.substring(0, 20) + "..." : node.title;
        ctx.fillText(label, node.x, node.y + node.radius + 18);
    });

    if (knowledgeMapState.showTopics) {
        orbitTopicNodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 10, 0, Math.PI * 2);
            ctx.fillStyle = "#fbbf24";
            ctx.fill();
            ctx.strokeStyle = "#2b2100";
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = "#f7f0d0";
            ctx.font = "10px Nunito, sans-serif";
            ctx.textAlign = "center";
            const label = node.label.length > 15 ? node.label.substring(0, 13) + "..." : node.label;
            ctx.fillText(label, node.x, node.y - 16);
        });
    }
    ctx.restore();

    knowledgeMapState.hoverTargets = [
        ...Object.entries(nodes).map(([docId, node]) => {
            const doc = knowledgeBase.documents?.[docId] || {};
            const point = worldToScreen(node.x, node.y, canvas);
            return {
                kind: "document",
                x: point.x,
                y: point.y,
                radius: node.radius * knowledgeMapState.scale,
                title: doc.title || node.title,
                meta: doc.source || doc.source_type || "document",
                summary: doc.summary || "No summary available yet.",
                concepts: Array.isArray(doc.concepts) ? doc.concepts.slice(0, 5) : [],
            };
        }),
        ...orbitTopicNodes.map(node => {
            const doc = knowledgeBase.documents?.[node.docId] || {};
            const point = worldToScreen(node.x, node.y, canvas);
            return {
                kind: "topic",
                x: point.x,
                y: point.y,
                radius: 12 * knowledgeMapState.scale,
                title: node.label,
                meta: doc.title || node.docId,
                summary: "Key topic extracted from this document.",
                concepts: [],
            };
        }),
    ];
    updateKnowledgeMapZoomLabel();
}

// ── Loading ────────────────────────────────────────────────────────────────

function showLoading(text = "Processing...") {
    document.getElementById("loading-text").textContent = text;
    document.getElementById("loading-overlay").classList.remove("hidden");
}

function hideLoading() {
    document.getElementById("loading-overlay").classList.add("hidden");
}

function buildOrbitTopicNodes(kb) {
    const topicNodes = [];
    Object.entries(kb.documents || {}).forEach(([docId, doc]) => {
        const concepts = Array.isArray(doc.concepts) ? doc.concepts.filter(Boolean).slice(0, 4) : [];
        concepts.forEach((concept, index) => {
            const total = Math.max(concepts.length, 1);
            topicNodes.push({
                id: `${docId}:${index}:${concept}`,
                docId,
                label: concept,
                normalized: normalizeTopic(concept),
                tokens: tokenizeTopic(concept),
                rank: index,
                angle: (-Math.PI / 2) + ((2 * Math.PI) / total) * index,
                x: 0,
                y: 0,
            });
        });
    });
    return topicNodes;
}

function buildTopicBridges(topicNodes) {
    const bridges = [];
    for (let i = 0; i < topicNodes.length; i += 1) {
        for (let j = i + 1; j < topicNodes.length; j += 1) {
            const from = topicNodes[i];
            const to = topicNodes[j];
            if (from.docId === to.docId) continue;
            const score = topicSimilarity(from, to);
            if (score < 0.52) continue;
            bridges.push({
                from,
                to,
                score,
                label: from.normalized === to.normalized ? from.label : `${from.label} ↔ ${to.label}`,
            });
        }
    }

    return bridges
        .sort((a, b) => b.score - a.score)
        .slice(0, 20);
}

function buildSharedConceptLinksFromTopicBridges(topicBridges) {
    const grouped = new Map();

    topicBridges.forEach(bridge => {
        const fromDoc = bridge.from.docId;
        const toDoc = bridge.to.docId;
        const key = [fromDoc, toDoc].sort().join("::");
        if (!grouped.has(key)) {
            grouped.set(key, {
                from: fromDoc,
                to: toDoc,
                concept: [],
                score: 0,
            });
        }

        const entry = grouped.get(key);
        entry.score = Math.max(entry.score, bridge.score);
        const label = bridge.from.normalized === bridge.to.normalized
            ? bridge.from.label
            : `${bridge.from.label} / ${bridge.to.label}`;
        if (!entry.concept.includes(label)) {
            entry.concept.push(label);
        }
    });

    return Array.from(grouped.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, 10)
        .map(entry => ({
            from: entry.from,
            to: entry.to,
            concept: entry.concept.slice(0, 3),
        }));
}

function normalizeTopic(topic) {
    return topic
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function tokenizeTopic(topic) {
    return new Set(
        normalizeTopic(topic)
            .split(" ")
            .filter(token => token && token.length > 2)
    );
}

function topicSimilarity(a, b) {
    if (!a.normalized || !b.normalized) return 0;
    if (a.normalized === b.normalized) return 1;
    if (a.normalized.includes(b.normalized) || b.normalized.includes(a.normalized)) return 0.86;

    const intersection = [...a.tokens].filter(token => b.tokens.has(token)).length;
    const union = new Set([...a.tokens, ...b.tokens]).size;
    const tokenScore = union ? intersection / union : 0;

    if (tokenScore > 0) return tokenScore;

    const wordA = a.normalized.split(" ").filter(Boolean);
    const wordB = b.normalized.split(" ").filter(Boolean);
    const stemOverlap = wordA.some(left => wordB.some(right =>
        left.startsWith(right.slice(0, Math.max(4, right.length - 2))) ||
        right.startsWith(left.slice(0, Math.max(4, left.length - 2)))
    ));

    return stemOverlap ? 0.58 : 0;
}

function renderTopicCorrelations(topicBridges, container) {
    if (!topicBridges.length) {
        container.innerHTML = '<p class="empty-state">Upload documents with overlapping ideas to see topic bridges.</p>';
        return;
    }

    const topBridges = topicBridges.slice(0, 10);
    container.innerHTML = topBridges.map(bridge => `
        <div class="topic-correlation-card">
            <div class="topic-correlation-title">
                <span>${escapeHtml(bridge.label)}</span>
                <span class="topic-correlation-count">${Math.round(bridge.score * 100)}% match</span>
            </div>
            <div class="topic-correlation-pair">
                <strong>${escapeHtml(resolveDocumentTitle(bridge.from.docId))}</strong> and
                <strong>${escapeHtml(resolveDocumentTitle(bridge.to.docId))}</strong>
                cover closely related topics.
            </div>
            <div class="topic-correlation-docs">
                <span class="topic-correlation-doc">${escapeHtml(bridge.from.label)}</span>
                <span class="topic-correlation-doc">${escapeHtml(bridge.to.label)}</span>
            </div>
        </div>
    `).join("");
}

function resolveDocumentTitle(docId) {
    const doc = documents.find(item => item.doc_id === docId);
    return doc ? doc.title : docId;
}

function setKnowledgeMapScale(nextScale) {
    const clamped = Math.min(knowledgeMapState.maxScale, Math.max(knowledgeMapState.minScale, nextScale));
    knowledgeMapState.scale = Math.round(clamped * 100) / 100;
    persistKnowledgeMapView();
    updateKnowledgeMapZoomLabel();
    if (knowledgeMapState.payload) {
        drawConnectionsMap();
    }
}

function updateKnowledgeMapZoomLabel() {
    const label = document.getElementById("connections-zoom-level");
    if (!label) return;
    label.textContent = `${Math.round(knowledgeMapState.scale * 100)}%`;
}

function persistKnowledgeMapView() {
    localStorage.setItem(KNOWLEDGE_MAP_STORAGE_KEY, JSON.stringify({
        scale: knowledgeMapState.scale,
        panX: knowledgeMapState.panX,
        panY: knowledgeMapState.panY,
        showSimilarity: knowledgeMapState.showSimilarity,
        showConcepts: knowledgeMapState.showConcepts,
        showTopics: knowledgeMapState.showTopics,
    }));
}

function restoreKnowledgeMapView() {
    try {
        const raw = localStorage.getItem(KNOWLEDGE_MAP_STORAGE_KEY);
        if (!raw) return;
        const saved = JSON.parse(raw);
        if (typeof saved.scale === "number") knowledgeMapState.scale = saved.scale;
        if (typeof saved.panX === "number") knowledgeMapState.panX = saved.panX;
        if (typeof saved.panY === "number") knowledgeMapState.panY = saved.panY;
        if (typeof saved.showSimilarity === "boolean") knowledgeMapState.showSimilarity = saved.showSimilarity;
        if (typeof saved.showConcepts === "boolean") knowledgeMapState.showConcepts = saved.showConcepts;
        if (typeof saved.showTopics === "boolean") knowledgeMapState.showTopics = saved.showTopics;
    } catch (_err) {
        // Ignore malformed local state.
    }
}

function toggleKnowledgeMapLayer(stateKey, buttonId) {
    knowledgeMapState[stateKey] = !knowledgeMapState[stateKey];
    persistKnowledgeMapView();
    syncKnowledgeMapToggleButtons();
    if (knowledgeMapState.payload) {
        drawConnectionsMap();
    }
}

function syncKnowledgeMapToggleButtons() {
    document.getElementById("toggle-similarity")?.classList.toggle("active", knowledgeMapState.showSimilarity);
    document.getElementById("toggle-concepts")?.classList.toggle("active", knowledgeMapState.showConcepts);
    document.getElementById("toggle-topics")?.classList.toggle("active", knowledgeMapState.showTopics);
}

function worldToScreen(x, y, canvas) {
    return {
        x: (x - canvas.width / 2) * knowledgeMapState.scale + canvas.width / 2 + knowledgeMapState.panX,
        y: (y - canvas.height / 2) * knowledgeMapState.scale + canvas.height / 2 + knowledgeMapState.panY,
    };
}

function findKnowledgeMapHoverTarget(event) {
    const canvas = document.getElementById("connections-canvas");
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    for (const target of knowledgeMapState.hoverTargets) {
        const dx = x - target.x;
        const dy = y - target.y;
        if ((dx * dx) + (dy * dy) <= target.radius * target.radius) {
            return target;
        }
    }

    return null;
}

function showConnectionsTooltip(target, clientX, clientY, tooltip) {
    const container = document.querySelector(".connections-container");
    const rect = container.getBoundingClientRect();
    const tagHtml = target.concepts.length
        ? `<div class="connections-tooltip-tags">${target.concepts.map(concept => `<span class="connections-tooltip-tag">${escapeHtml(concept)}</span>`).join("")}</div>`
        : "";

    tooltip.innerHTML = `
        <div class="connections-tooltip-title">${escapeHtml(target.title)}</div>
        <div class="connections-tooltip-meta">${escapeHtml(target.meta)}</div>
        <div class="connections-tooltip-body">${escapeHtml(target.summary)}</div>
        ${tagHtml}
    `;
    tooltip.classList.remove("hidden");

    const offset = 14;
    let left = clientX - rect.left + offset;
    let top = clientY - rect.top + offset;

    if (left + tooltip.offsetWidth > rect.width - 8) {
        left = Math.max(8, clientX - rect.left - tooltip.offsetWidth - offset);
    }
    if (top + tooltip.offsetHeight > rect.height - 8) {
        top = Math.max(8, clientY - rect.top - tooltip.offsetHeight - offset);
    }

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
}

function hideConnectionsTooltip() {
    const tooltip = document.getElementById("connections-tooltip");
    if (!tooltip) return;
    tooltip.classList.add("hidden");
}

// ── Helpers ────────────────────────────────────────────────────────────────

function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
