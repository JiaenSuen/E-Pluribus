 
const form = document.querySelector("#analysisForm");
const imageInput = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const previewCard = document.querySelector("#previewCard");
const imagePreview = document.querySelector("#imagePreview");
const fileName = document.querySelector("#fileName");

const submitButton = document.querySelector("#submitButton");
const emptyState = document.querySelector("#emptyState");
const loadingCard = document.querySelector("#loadingCard");
const answerCard = document.querySelector("#answerCard");
const finalAnswer = document.querySelector("#finalAnswer");
const requestId = document.querySelector("#requestId");

const traceSection = document.querySelector("#traceSection");
const traceList = document.querySelector("#traceList");
const traceMeta = document.querySelector("#traceMeta");

const documentsSection = document.querySelector("#documentsSection");
const documentList = document.querySelector("#documentList");

const loadModulesButton = document.querySelector("#loadModulesButton");
const moduleDialog = document.querySelector("#moduleDialog");
const moduleList = document.querySelector("#moduleList");
const closeDialog = document.querySelector("#closeDialog");


/* =========================================================
   Basic visibility helpers
========================================================= */

function show(element) {
  if (!element) return;
  element.classList.remove("is-hidden");
}


function hide(element) {
  if (!element) return;
  element.classList.add("is-hidden");
}


/* =========================================================
   Reset and loading state
========================================================= */

function clearPreviousResult() {
  finalAnswer.textContent = "";
  requestId.textContent = "";

  traceList.innerHTML = "";
  traceMeta.textContent = "";

  documentList.innerHTML = "";

  hide(answerCard);
  hide(traceSection);
  hide(documentsSection);

   
  if (traceSection instanceof HTMLDetailsElement) {
    traceSection.open = false;
  }
}


function setLoading(isLoading) {
  submitButton.disabled = isLoading;

  if (isLoading) {
    clearPreviousResult();

    hide(emptyState);
    show(loadingCard);

    submitButton.querySelector("span:first-child").textContent =
      "Running analysis";
  } else {
    hide(loadingCard);

    submitButton.querySelector("span:first-child").textContent =
      "Run analysis";
  }
}


/* =========================================================
   Pipeline trace
========================================================= */

function renderTrace(trace = []) {
  traceList.innerHTML = "";
  traceMeta.textContent = "";

  if (!Array.isArray(trace) || trace.length === 0) {
    hide(traceSection);
    return;
  }

  traceMeta.textContent =
    `${trace.length} ${trace.length === 1 ? "stage" : "stages"}`;

  trace.forEach((stage, index) => {
    const card = document.createElement("article");
    card.className = "trace-card";

    const step = document.createElement("div");
    step.className = "step-index";
    step.textContent = index + 1;

    const body = document.createElement("div");

    const title = document.createElement("h3");
    title.textContent =
      stage.module_name ||
      stage.module_id ||
      `Pipeline stage ${index + 1}`;

    const subtitle = document.createElement("p");
    subtitle.className = "muted";
    subtitle.textContent =
      stage.module_id || "Unknown module";

    const pre = document.createElement("pre");
    pre.className = "trace-data";

    try {
      pre.textContent = JSON.stringify(stage.data ?? {}, null, 2);
    } catch (error) {
      pre.textContent = String(stage.data ?? "No stage data.");
    }

    body.append(title, subtitle, pre);

    const status = document.createElement("span");

    const isSuccessful =
      String(stage.status || "").toLowerCase() === "ok";

    status.className =
      isSuccessful ? "status-ok" : "status-error";

    const elapsed =
      typeof stage.elapsed_ms === "number"
        ? `${stage.elapsed_ms} ms`
        : "time unavailable";

    status.textContent =
      `${stage.status || "unknown"} · ${elapsed}`;

    card.append(step, body, status);
    traceList.appendChild(card);
  });

  show(traceSection);

 
  if (traceSection instanceof HTMLDetailsElement) {
    traceSection.open = false;
  }
}


/* =========================================================
   Evidence documents
========================================================= */

function normalizeScore(value) {
  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return 0;
  }
 
  const percentage =
    numericValue <= 1
      ? numericValue * 100
      : numericValue;

  return Math.min(100, Math.max(0, percentage));
}


function renderDocuments(result) {
  documentList.innerHTML = "";

  const docs =
    result?.evidence_package?.top_documents || [];

  if (!Array.isArray(docs) || docs.length === 0) {
    hide(documentsSection);
    return;
  }

  docs.forEach((doc, index) => {
    const card = document.createElement("article");
    card.className = "document-card";

   
    const header = document.createElement("div");
    header.className = "document-card-header";

    const title = document.createElement("h4");
    title.className = "document-title";
    title.textContent =
      doc.title || `Evidence document ${index + 1}`;

    const rawScore =
      doc.contextual_relevance ??
      doc.contextual_relevance_score ??
      doc.relevance_score ??
      doc.score ??
      0;

    const scorePercentage = normalizeScore(rawScore);

    const scoreLabel = document.createElement("span");
    scoreLabel.className = "document-score";
    scoreLabel.textContent = `${Math.round(scorePercentage)}%`;
    scoreLabel.title = "Contextual relevance score";

    header.append(title, scoreLabel);

    /*
      Metadata
    */
    const meta = document.createElement("div");
    meta.className = "document-meta";

    const source =
      doc.source ||
      doc.journal ||
      doc.publisher ||
      "Unknown source";

    const year =
      doc.year ||
      "n.d.";

    const doi =
      doc.doi ||
      "no DOI";

    meta.textContent = `${source} · ${year} · ${doi}`;

    /*
      Score bar
    */
    const scoreBar = document.createElement("div");
    scoreBar.className = "score-bar";
    scoreBar.setAttribute(
      "aria-label",
      `Contextual relevance: ${Math.round(scorePercentage)} percent`
    );

    const scoreFill = document.createElement("div");
    scoreFill.className = "score-fill";
    scoreFill.style.setProperty(
      "--score-width",
      `${scorePercentage}%`
    );

    scoreBar.appendChild(scoreFill);

    /*
      Abstract
    */
    const abstract = document.createElement("p");
    abstract.className = "document-abstract";
    abstract.textContent =
      doc.abstract ||
      doc.summary ||
      "No abstract is available for this document.";

    /*
      Keywords
    */
    const keywords = document.createElement("div");
    keywords.className = "keyword-row";

    const keywordValues =
      Array.isArray(doc.keywords)
        ? doc.keywords
        : [];

  
    keywordValues.slice(0, 6).forEach((keyword) => {
      const tag = document.createElement("span");
      tag.className = "keyword";
      tag.textContent = String(keyword);
      keywords.appendChild(tag);
    });

    card.append(
      header,
      meta,
      scoreBar,
      abstract
    );

    if (keywordValues.length > 0) {
      card.appendChild(keywords);
    }

    documentList.appendChild(card);
  });

  show(documentsSection);
}


/* =========================================================
   Final result
========================================================= */

function renderResult(result) {
  const answer =
    result?.final_answer ||
    "No answer returned.";

  finalAnswer.textContent = answer;

  requestId.textContent =
    result?.request?.id
      ? `request ${result.request.id}`
      : "";

  renderTrace(result?.trace || []);
  renderDocuments(result);

  show(answerCard);
}


/* =========================================================
   Image preview
========================================================= */

function setImagePreview(file) {
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    window.alert("Please select a valid image file.");
    imageInput.value = "";
    hide(previewCard);
    return;
  }

  const reader = new FileReader();

  reader.onload = (event) => {
    imagePreview.src = event.target.result;
    fileName.textContent = file.name;
    show(previewCard);
  };

  reader.onerror = () => {
    imageInput.value = "";
    hide(previewCard);
    window.alert("The selected image could not be read.");
  };

  reader.readAsDataURL(file);
}


imageInput.addEventListener("change", () => {
  setImagePreview(imageInput.files[0]);
});


["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();

    dropZone.classList.add("is-dragging");
  });
});


["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();

    dropZone.classList.remove("is-dragging");
  });
});


dropZone.addEventListener("drop", (event) => {
  const files = event.dataTransfer?.files;

  if (!files || files.length === 0) {
    return;
  }

  const file = files[0];

  if (!file.type.startsWith("image/")) {
    window.alert("Please drop a valid image file.");
    return;
  }

  
  try {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;
  } catch (error) {
  
    console.warn(
      "Could not synchronize dropped file with file input.",
      error
    );
  }

  setImagePreview(file);
});


/* =========================================================
   Analysis request
========================================================= */

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!imageInput.files.length) {
    window.alert("Please select an ecological field image.");
    return;
  }

  const formData = new FormData(form);

  setLoading(true);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    let payload;

    try {
      payload = await response.json();
    } catch (error) {
      throw new Error(
        "The server returned an invalid response."
      );
    }

    if (!response.ok) {
      throw new Error(
        payload?.error ||
        payload?.message ||
        "Pipeline request failed."
      );
    }

    renderResult(payload);
  } catch (error) {
    finalAnswer.textContent =
      error instanceof Error
        ? error.message
        : "An unexpected error occurred.";

    requestId.textContent = "error";

    hide(traceSection);
    hide(documentsSection);
    show(answerCard);
  } finally {
    setLoading(false);
  }
});


/* =========================================================
   Active modules dialog
========================================================= */

loadModulesButton.addEventListener("click", async () => {
  moduleList.innerHTML = "";

  loadModulesButton.disabled = true;
  loadModulesButton.textContent = "Loading modules";

  try {
    const response = await fetch("/api/modules");

    let payload;

    try {
      payload = await response.json();
    } catch (error) {
      throw new Error(
        "The module endpoint returned an invalid response."
      );
    }

    if (!response.ok) {
      throw new Error(
        payload?.error ||
        "Unable to load active modules."
      );
    }

    const modules =
      Array.isArray(payload?.modules)
        ? payload.modules
        : [];

    if (modules.length === 0) {
      const emptyMessage = document.createElement("p");
      emptyMessage.className = "muted";
      emptyMessage.textContent =
        "No active modules were returned.";

      moduleList.appendChild(emptyMessage);
    }

    modules.forEach((module, index) => {
      const card = document.createElement("article");
      card.className = "module-card";

      const title = document.createElement("h3");

      const order =
        module.order ??
        index + 1;

      title.textContent =
        `${order}. ${module.name || "Unnamed module"}`;

      const meta = document.createElement("p");
      meta.textContent =
        `${module.id || "unknown-id"} · ${module.source || "unknown source"}`;

      const description = document.createElement("p");
      description.textContent =
        module.description ||
        "No description.";

      card.append(
        title,
        meta,
        description
      );

      moduleList.appendChild(card);
    });

    moduleDialog.showModal();
  } catch (error) {
    const errorMessage = document.createElement("p");
    errorMessage.className = "status-error";
    errorMessage.textContent =
      error instanceof Error
        ? error.message
        : "Unable to load active modules.";

    moduleList.appendChild(errorMessage);
    moduleDialog.showModal();
  } finally {
    loadModulesButton.disabled = false;
    loadModulesButton.textContent = "View active modules";
  }
});


closeDialog.addEventListener("click", () => {
  moduleDialog.close();
});


moduleDialog.addEventListener("click", (event) => {
 
  if (event.target === moduleDialog) {
    moduleDialog.close();
  }
});
 
