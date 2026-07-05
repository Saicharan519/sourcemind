import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_URL,
  timeout: 60_000,
});

// -- Sources ---------------------------------------------------------------

export const listSources = async () => {
  const { data } = await api.get("/api/sources");
  return data;
};

export const getSource = async (sourceId) => {
  const { data } = await api.get(`/api/sources/${sourceId}`);
  return data;
};

export const deleteSource = async (sourceId) => {
  const { data } = await api.delete(`/api/sources/${sourceId}`);
  return data;
};

// -- Ingestion -------------------------------------------------------------

export const ingestDocument = async (file) => {
  const fd = new FormData();
  fd.append("file", file);
  const { data } = await api.post("/api/ingest/document", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const ingestVideo = async (videoUrl) => {
  const { data } = await api.post("/api/ingest/video", {
    youtube_url: videoUrl,
  });
  return data;
};

export const ingestVideoFile = async (file) => {
  const fd = new FormData();
  fd.append("file", file);
  const { data } = await api.post("/api/ingest/video-file", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

// -- Evaluation ------------------------------------------------------------

export const triggerEvaluation = async (sourceId) => {
  const { data } = await api.post(`/api/evaluate/${sourceId}`);
  return data;
};

export const getEvaluation = async (sourceId) => {
  const { data } = await api.get(`/api/evaluate/${sourceId}`);
  return data;
};

// -- Streaming chat URL builders -------------------------------------------

export const buildChatStreamUrl = ({ sourceId, question, history }) => {
  const params = new URLSearchParams({
    source_id: sourceId,
    question,
  });
  if (history && history.length > 0) {
    params.set("history", JSON.stringify(history));
  }
  return `${API_URL}/api/chat/stream?${params.toString()}`;
};

export const buildGlobalChatStreamUrl = ({ question, history }) => {
  const params = new URLSearchParams({ question });
  if (history && history.length > 0) {
    params.set("history", JSON.stringify(history));
  }
  return `${API_URL}/api/chat/global/stream?${params.toString()}`;
};
