/* ── School Helper Dashboard App ───────────────────────────────────────── */

const API = "";  // same origin

// ── State ──────────────────────────────────────────────────────────────────

let documents = [];
let courses = [];
let activeCourseId = null;
let sessionId = sessionStorage.getItem("sh_session") || crypto.randomUUID();
sessionStorage.setItem("sh_session", sessionId);

// ── Helpers ────────────────────────────────────────────────────────────────

function courseApi(path) {
    return `${API}/api/courses/${activeCourseId}${path}`;
}

function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ── Init ───────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    initRouter();
    initUpload();
    initSearch();
    initChat();
    initConnections();
    initCourseModal();
    initChatWidget();
    loadCourses();
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
    }

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            window.location.hash = page;
            navigate(page);
        });
    });

    const hash = window.location.hash.slice(1) || "documents";
    navigate(hash);

    window.addEventListener("hashchange", () => {
        navigate(window.location.hash.slice(1) || "documents");
    });
}

// ── Course Management ─────────────────────────────────────────────────────

async function loadCourses() {
    try {
        const res = await fetch(`${API}/api/courses`);
        const data = await res.json();
        courses = data.courses || [];
    } catch (err) {
        console.error("Failed to load courses:", err);
        courses = [];
    }

    if (courses.length === 0) {
        activeCourseId = null;
        showNoCoursePrompt();
    } else {
        // Restore last active or pick first
        const stored = sessionStorage.getItem("sh_active_course");
        const valid = courses.find(c => c.id === stored);
        activeCourseId = valid ? stored : courses[0].id;
        hideNoCoursePrompt();
        loadCourseData();
    }
    renderCourseTabs();
}

function renderCourseTabs() {
    const container = document.getElementById("course-tabs");
    container.innerHTML = courses.map(c => `
        <button class="course-tab ${c.id === activeCourseId ? 'active' : ''}"
                data-id="${c.id}"
                style="border-bottom-color: ${c.id === activeCourseId ? c.color : 'transparent'}">
            ${escapeHtml(c.name)}
            <span class="course-tab-delete" data-id="${c.id}" title="Delete course">&times;</span>
        </button>
    `).join("");

    // Tab click -> switch course
    container.querySelectorAll(".course-tab").forEach(tab => {
        tab.addEventListener("click", (e) => {
            if (e.target.classList.contains("course-tab-delete")) return;
            setActiveCourse(tab.dataset.id);
        });
    });

    // Delete click
    container.querySelectorAll(".course-tab-delete").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            deleteCourse(btn.dataset.id);
        });
    });
}

function setActiveCourse(courseId) {
    activeCourseId = courseId;
    sessionStorage.setItem("sh_active_course", courseId);
    renderCourseTabs();
    updateWidgetCourseName();
    clearWidgetMessages();
    loadCourseData();
}

function loadCourseData() {
    if (!activeCourseId) return;
    loadDocuments();
    loadStats();
    // Re-render the active page if needed
    const hash = window.location.hash.slice(1) || "documents";
    if (hash === "connections") renderConnections();
    if (hash === "summaries") renderSummaries();

}

async function deleteCourse(courseId) {
    const course = courses.find(c => c.id === courseId);
    const name = course ? course.name : courseId;
    if (!confirm(`Delete course "${name}" and all its documents?`)) return;

    try {
        const res = await fetch(`${API}/api/courses/${courseId}`, { method: "DELETE" });
        if (!res.ok) throw new Error("Delete failed");
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
    loadCourses();
}

function showNoCoursePrompt() {
    document.getElementById("no-course-prompt").classList.remove("hidden");
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    document.querySelector(".search-bar").style.display = "none";
}

function hideNoCoursePrompt() {
    document.getElementById("no-course-prompt").classList.add("hidden");
    document.querySelector(".search-bar").style.display = "";
    // Restore active page
    const hash = window.location.hash.slice(1) || "documents";
    const pageEl = document.getElementById(`page-${hash}`);
    if (pageEl) pageEl.classList.add("active");
}

// ── Course Modal ──────────────────────────────────────────────────────────

function initCourseModal() {
    const modal = document.getElementById("course-modal");
    const nameInput = document.getElementById("course-name-input");
    let selectedColor = "#6366f1";

    // Open modal
    document.getElementById("course-add-btn").addEventListener("click", () => openCourseModal());
    document.getElementById("no-course-create-btn").addEventListener("click", () => openCourseModal());

    function openCourseModal() {
        nameInput.value = "";
        selectedColor = "#6366f1";
        document.querySelectorAll(".color-swatch").forEach(s => {
            s.classList.toggle("active", s.dataset.color === selectedColor);
        });
        modal.classList.remove("hidden");
        nameInput.focus();
    }

    // Color selection
    document.getElementById("color-options").addEventListener("click", (e) => {
        const swatch = e.target.closest(".color-swatch");
        if (!swatch) return;
        selectedColor = swatch.dataset.color;
        document.querySelectorAll(".color-swatch").forEach(s => {
            s.classList.toggle("active", s.dataset.color === selectedColor);
        });
    });

    // Cancel
    document.getElementById("course-modal-cancel").addEventListener("click", () => {
        modal.classList.add("hidden");
    });

    // Create
    document.getElementById("course-modal-create").addEventListener("click", async () => {
        const name = nameInput.value.trim();
        if (!name) { nameInput.focus(); return; }

        try {
            const res = await fetch(`${API}/api/courses`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, color: selectedColor }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Failed to create course");
            modal.classList.add("hidden");
            // Reload courses and switch to new one
            await loadCourses();
            setActiveCourse(data.course_id);
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    });

    // Enter key in name input
    nameInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") document.getElementById("course-modal-create").click();
    });

    // Click outside to close
    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.classList.add("hidden");
    });
}

// ── Upload ─────────────────────────────────────────────────────────────────

function initUpload() {
    const tabs = document.querySelectorAll(".upload-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            document.querySelectorAll(".upload-content").forEach(c => c.classList.add("hidden"));
            document.getElementById(`tab-${tab.dataset.tab}`).classList.remove("hidden");
        });
    });

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

    document.getElementById("url-btn").addEventListener("click", addURL);
    document.getElementById("url-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") addURL();
    });

    document.getElementById("text-btn").addEventListener("click", addText);
}

async function uploadPDFs(files) {
    if (!activeCourseId) return;
    for (const file of files) {
        showLoading(`Uploading ${file.name}...`);
        try {
            const form = new FormData();
            form.append("file", file);
            const res = await fetch(courseApi("/documents/upload-pdf"), { method: "POST", body: form });
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
    if (!activeCourseId) return;
    const input = document.getElementById("url-input");
    const url = input.value.trim();
    if (!url) return;

    showLoading("Fetching and indexing URL...");
    try {
        const res = await fetch(courseApi("/documents/add-url"), {
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
    if (!activeCourseId) return;
    const titleInput = document.getElementById("text-title");
    const contentInput = document.getElementById("text-content");
    const title = titleInput.value.trim();
    const text = contentInput.value.trim();
    if (!title || !text) { alert("Please provide both a title and text content."); return; }

    showLoading("Indexing text...");
    try {
        const res = await fetch(courseApi("/documents/add-text"), {
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
    if (!activeCourseId) return;
    try {
        const res = await fetch(courseApi("/documents"));
        const data = await res.json();
        documents = data.documents || [];
        renderDocuments();
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
    if (!activeCourseId) return;
    if (!confirm("Delete this document and all its data?")) return;

    try {
        const res = await fetch(courseApi(`/documents/${docId}`), { method: "DELETE" });
        if (!res.ok) throw new Error("Delete failed");
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
    loadDocuments();
    loadStats();
}

// ── Stats ──────────────────────────────────────────────────────────────────

async function loadStats() {
    if (!activeCourseId) {
        document.getElementById("stat-docs").textContent = "0";
        document.getElementById("stat-chunks").textContent = "0";
        document.getElementById("stat-qa").textContent = "0";
        return;
    }
    try {
        const res = await fetch(courseApi("/stats"));
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
    if (!activeCourseId) return;
    query = query.trim();
    const results = document.getElementById("search-results");
    if (!query) { results.classList.add("hidden"); return; }

    try {
        const res = await fetch(courseApi("/search"), {
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

// ── Chat (full page) ──────────────────────────────────────────────────────

function initChat() {
    const input = document.getElementById("chat-input");
    const btn = document.getElementById("chat-send");

    btn.addEventListener("click", () => sendChat());
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChat();
    });
}

async function sendChat() {
    if (!activeCourseId) return;
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) return;

    input.value = "";
    appendChatMsg("user", message);

    const typingEl = appendChatMsg("assistant", "Thinking...");
    typingEl.style.opacity = "0.5";

    try {
        const res = await fetch(courseApi("/chat"), {
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

function toggleSummary(header) {
    const body = header.nextElementSibling;
    const toggle = header.querySelector(".summary-toggle");
    body.classList.toggle("open");
    toggle.classList.toggle("open");
}

// ── Connections / Knowledge Map ────────────────────────────────────────────

function initConnections() {
    document.getElementById("refresh-connections-btn").addEventListener("click", async () => {
        if (!activeCourseId) return;
        showLoading("Computing knowledge connections...");
        try {
            const res = await fetch(courseApi("/connections/refresh"), { method: "POST" });
            if (!res.ok) throw new Error("Refresh failed");
            await loadDocuments();
            renderConnections();
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
        hideLoading();
    });
}

async function renderConnections() {
    if (!activeCourseId) return;
    const canvas = document.getElementById("connections-canvas");
    const emptyEl = document.getElementById("connections-empty");
    const ctx = canvas.getContext("2d");

    let connections = [];
    try {
        const res = await fetch(courseApi("/connections"));
        const data = await res.json();
        connections = data.connections || [];
    } catch (err) {
        console.error("Failed to load connections:", err);
    }

    if (!connections.length || documents.length < 2) {
        canvas.style.display = "none";
        emptyEl.style.display = "block";
        return;
    }

    canvas.style.display = "block";
    emptyEl.style.display = "none";

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = Math.max(rect.height, 400);

    const nodes = {};
    documents.forEach((doc, i) => {
        const angle = (2 * Math.PI * i) / documents.length - Math.PI / 2;
        const rx = canvas.width * 0.35;
        const ry = canvas.height * 0.35;
        nodes[doc.doc_id] = {
            x: canvas.width / 2 + rx * Math.cos(angle),
            y: canvas.height / 2 + ry * Math.sin(angle),
            title: doc.title,
            type: doc.source_type,
        };
    });

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const seen = new Set();
    connections.forEach(conn => {
        const key = [conn.from_doc_id, conn.to_doc_id].sort().join("-");
        if (seen.has(key)) return;
        seen.add(key);

        const from = nodes[conn.from_doc_id];
        const to = nodes[conn.to_doc_id];
        if (!from || !to) return;

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.strokeStyle = `rgba(99, 102, 241, ${Math.min(conn.similarity, 0.8)})`;
        ctx.lineWidth = Math.max(1, conn.similarity * 4);
        ctx.stroke();

        const mx = (from.x + to.x) / 2;
        const my = (from.y + to.y) / 2;
        ctx.fillStyle = "rgba(156, 163, 175, 0.7)";
        ctx.font = "10px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText((conn.similarity * 100).toFixed(0) + "%", mx, my - 4);
    });

    const typeColors = { pdf: "#f87171", url: "#60a5fa", text: "#4ade80" };

    Object.values(nodes).forEach(node => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, 20, 0, Math.PI * 2);
        ctx.fillStyle = typeColors[node.type] || "#6366f1";
        ctx.fill();
        ctx.strokeStyle = "#1c1f2e";
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.fillStyle = "#e4e4e7";
        ctx.font = "12px Inter, sans-serif";
        ctx.textAlign = "center";
        const label = node.title.length > 20 ? node.title.substring(0, 18) + "..." : node.title;
        ctx.fillText(label, node.x, node.y + 34);
    });
}

// ── Loading ────────────────────────────────────────────────────────────────

function showLoading(text = "Processing...") {
    document.getElementById("loading-text").textContent = text;
    document.getElementById("loading-overlay").classList.remove("hidden");
}

function hideLoading() {
    document.getElementById("loading-overlay").classList.add("hidden");
}

// ── Floating Chat Widget ──────────────────────────────────────────────────

function initChatWidget() {
    const toggle = document.getElementById("chat-widget-toggle");
    const panel = document.getElementById("chat-widget-panel");
    const close = document.getElementById("chat-widget-close");
    const input = document.getElementById("chat-widget-input");
    const send = document.getElementById("chat-widget-send");

    toggle.addEventListener("click", () => {
        panel.classList.toggle("hidden");
        if (!panel.classList.contains("hidden")) {
            input.focus();
            updateWidgetCourseName();
        }
    });

    close.addEventListener("click", () => {
        panel.classList.add("hidden");
    });

    send.addEventListener("click", () => sendWidgetChat());
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendWidgetChat();
    });
}

function updateWidgetCourseName() {
    const label = document.getElementById("chat-widget-course-name");
    if (activeCourseId) {
        const course = courses.find(c => c.id === activeCourseId);
        label.textContent = course ? `Chat — ${course.name}` : "Course Chat";
    } else {
        label.textContent = "Course Chat";
    }
}

function clearWidgetMessages() {
    const container = document.getElementById("chat-widget-messages");
    container.innerHTML = '<div class="chat-welcome"><p>Ask about your course materials...</p></div>';
}

async function sendWidgetChat() {
    if (!activeCourseId) return;
    const input = document.getElementById("chat-widget-input");
    const message = input.value.trim();
    if (!message) return;

    input.value = "";
    appendWidgetMsg("user", message);

    const typingEl = appendWidgetMsg("assistant", "Thinking...");
    typingEl.style.opacity = "0.5";

    try {
        const res = await fetch(courseApi("/chat"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: `widget_${sessionId}` }),
        });
        const data = await res.json();

        typingEl.remove();
        appendWidgetAnswer(data);
        loadStats();
    } catch (err) {
        typingEl.remove();
        appendWidgetMsg("assistant", `Error: ${err.message}`);
    }
}

function appendWidgetMsg(role, text) {
    const container = document.getElementById("chat-widget-messages");
    const welcome = container.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const div = document.createElement("div");
    div.className = `chat-msg ${role}`;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

function appendWidgetAnswer(data) {
    const container = document.getElementById("chat-widget-messages");

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

    div.innerHTML = html;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}
